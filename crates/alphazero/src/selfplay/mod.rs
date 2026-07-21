use super::batcher::{Batcher, BatcherClient};
use search::{GumbelSearchProfile, Mcts, MctsConfig, MctsVariant, SearchResult};
use super::replay::{ReplayBuffer, Transition};
use super::representation::AlphaZeroRepresentation;
use super::RepresentedEvaluator;
use search::NoExtraRules;
use engine_core::agent::PolicyMode;
use engine_core::game::{GameState, TerminalValue};
use rand::distr::Distribution;
use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct SelfPlayStats {
    pub games: usize,
    pub moves: usize,
    pub full_searches: usize,
    pub fast_searches: usize,
    pub resignations: usize,
    pub tt_hits: u64,
    pub tt_misses: u64,
    pub tt_inserts: u64,
}

struct CompletedGame {
    stats: SelfPlayStats,
    trajectory: Vec<Transition>,
}

impl SelfPlayStats {
    pub fn avg_moves_per_game(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.moves as f64 / self.games as f64
        }
    }

    fn add(&mut self, other: SelfPlayStats) {
        self.games += other.games;
        self.moves += other.moves;
        self.full_searches += other.full_searches;
        self.fast_searches += other.fast_searches;
        self.resignations += other.resignations;
        self.tt_hits += other.tt_hits;
        self.tt_misses += other.tt_misses;
        self.tt_inserts += other.tt_inserts;
    }
}

#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    pub num_games: usize,
    pub threads: usize,
    /// Games longer than this are truncated and scored as draws.
    pub max_moves: usize,
    /// Moves sampled from the visit distribution before switching to argmax.
    ///
    /// This is retained for legacy self-play. Chess AZ v2 uses
    /// `chess_v2_temperature` instead.
    pub temperature_moves: usize,
    /// Chess AZ v2 move-selection schedule. `None` preserves the legacy
    /// temperature behavior, including Gumbel's historical argmax selection.
    pub chess_v2_temperature: Option<SelfPlayTemperature>,
    /// Print self-play progress every N completed games. Set 0 to disable.
    pub progress_every: usize,
    /// Maximum chess NN evaluation cache entries for one self-play iteration.
    /// Set to 0 to disable the cache.
    pub tt_entries: usize,
    pub fast_simulations: usize,
    pub full_simulation_probability: f32,
    /// Paired Gumbel profiles for Chess AZ v2 self-play. `None` keeps the
    /// legacy single-budget configuration above.
    pub chess_v2_gumbel_profiles: Option<ChessV2GumbelProfiles>,
    pub resignation_enabled: bool,
    pub resignation_threshold: f32,
    pub resignation_consecutive_moves: usize,
    pub resignation_min_ply: usize,
    pub resignation_disable_probability: f32,
    pub mcts: MctsConfig,
}

/// Temperature schedule for sampling an improved root policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfPlayTemperature {
    /// Number of initial plies sampled at temperature 1.0.
    pub full_temperature_plies: usize,
    /// Number of following plies sampled at `reduced_temperature`.
    pub reduced_temperature_plies: usize,
    pub reduced_temperature: f32,
}

impl Default for SelfPlayTemperature {
    fn default() -> Self {
        Self {
            full_temperature_plies: 20,
            reduced_temperature_plies: 20,
            reduced_temperature: 0.5,
        }
    }
}

impl SelfPlayTemperature {
    pub fn validate(self) -> Result<(), &'static str> {
        if !self.reduced_temperature.is_finite() || self.reduced_temperature <= 0.0 {
            return Err("reduced self-play temperature must be finite and positive");
        }
        Ok(())
    }

    pub fn at_ply(self, ply: usize) -> Option<f32> {
        if ply < self.full_temperature_plies {
            Some(1.0)
        } else if ply < self.full_temperature_plies + self.reduced_temperature_plies {
            Some(self.reduced_temperature)
        } else {
            None
        }
    }
}

/// The randomized search budgets used by Chess AZ v2 self-play.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChessV2GumbelProfiles {
    pub full: GumbelSearchProfile,
    pub fast: GumbelSearchProfile,
}

impl ChessV2GumbelProfiles {
    pub const DEFAULT: Self = Self {
        full: GumbelSearchProfile::new(128, 16),
        fast: GumbelSearchProfile::new(64, 8),
    };

    pub fn validate(self) -> Result<(), &'static str> {
        self.full.validate()?;
        self.fast.validate()
    }
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            num_games: 100,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            max_moves: 256,
            temperature_moves: 30,
            chess_v2_temperature: None,
            progress_every: 25,
            tt_entries: 1_000_000,
            fast_simulations: 100,
            full_simulation_probability: 0.25,
            chess_v2_gumbel_profiles: None,
            resignation_enabled: true,
            resignation_threshold: -0.95,
            resignation_consecutive_moves: 3,
            resignation_min_ply: 60,
            resignation_disable_probability: 0.10,
            mcts: MctsConfig::default(),
        }
    }
}

impl SelfPlayConfig {
    /// Defaults for the new chess-only AlphaZero v2 format. This is separate
    /// from `Default` so existing generic and legacy chess callers keep their
    /// previous self-play behavior.
    pub fn chess_v2_defaults() -> Self {
        Self {
            temperature_moves: 0,
            chess_v2_temperature: Some(SelfPlayTemperature::default()),
            fast_simulations: ChessV2GumbelProfiles::DEFAULT.fast.simulations,
            full_simulation_probability: 0.5,
            chess_v2_gumbel_profiles: Some(ChessV2GumbelProfiles::DEFAULT),
            mcts: MctsConfig {
                simulations: ChessV2GumbelProfiles::DEFAULT.full.simulations,
                variant: MctsVariant::Gumbel {
                    sampled_actions: ChessV2GumbelProfiles::DEFAULT.full.root_candidates,
                },
                ..MctsConfig::default()
            },
            ..Self::default()
        }
    }

    /// Validates the optional Chess AZ v2 self-play additions. Legacy callers
    /// may leave both fields as `None`.
    pub fn validate_chess_v2(&self) -> Result<(), &'static str> {
        if let Some(temperature) = self.chess_v2_temperature {
            temperature.validate()?;
        }
        if let Some(profiles) = self.chess_v2_gumbel_profiles {
            profiles.validate()?;
            if !matches!(self.mcts.variant, MctsVariant::Gumbel { .. }) {
                return Err("Chess AZ v2 Gumbel profiles require Gumbel MCTS");
            }
            if !self.full_simulation_probability.is_finite()
                || !(0.0..=1.0).contains(&self.full_simulation_probability)
            {
                return Err("full simulation probability must be finite and in [0, 1]");
            }
        }
        Ok(())
    }
}

/// Plays `cfg.num_games` games of self-play across `cfg.threads` threads, all
/// sharing `batcher` for network evaluation, appending trajectories to `replay`.
pub fn self_play<G, Rep>(
    batcher: &Batcher,
    replay: &ReplayBuffer,
    cfg: &SelfPlayConfig,
) -> SelfPlayStats
where
    G: GameState + Clone + Default,
    Rep: AlphaZeroRepresentation<G> + Default,
{
    let finished = AtomicUsize::new(0);
    let started = Instant::now();
    let stats = Arc::new(Mutex::new(SelfPlayStats::default()));

    std::thread::scope(|scope| {
        for _ in 0..cfg.threads.max(1) {
            let stats = Arc::clone(&stats);
            let finished = &finished;
            scope.spawn(move || {
                let representation = Rep::default();
                let evaluator = RepresentedEvaluator::new(representation.clone(), batcher.client());
                let mut mcts = Mcts::new(evaluator, cfg.mcts, NoExtraRules);
                while finished.load(Ordering::Relaxed) < cfg.num_games {
                    let Some(completed) =
                        play_game::<G, Rep>(&mut mcts, &representation, cfg, || {
                            finished.load(Ordering::Relaxed) >= cfg.num_games
                        })
                    else {
                        break;
                    };
                    let Ok(done_before) = claim_completed_game(finished, cfg.num_games) else {
                        break;
                    };
                    replay.add(completed.trajectory);
                    stats.lock().unwrap().add(completed.stats);
                    let done = done_before + 1;
                    maybe_print_progress(done, cfg, &stats, started);
                }
            });
        }
    });
    Arc::try_unwrap(stats).unwrap().into_inner().unwrap()
}

fn claim_completed_game(finished: &AtomicUsize, target: usize) -> Result<usize, usize> {
    finished.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |done| {
        (done < target).then_some(done + 1)
    })
}

fn maybe_print_progress(
    done: usize,
    cfg: &SelfPlayConfig,
    stats: &Arc<Mutex<SelfPlayStats>>,
    started: Instant,
) {
    let every = cfg.progress_every;
    if every == 0 || (done != cfg.num_games && !done.is_multiple_of(every)) {
        return;
    }
    let elapsed = started.elapsed().as_secs_f64().max(1e-6);
    let stats = stats.lock().unwrap();
    println!(
        "self-play progress: {done}/{} games, {:.1} games/s, {:.1} positions/s, {:.1} moves/game",
        cfg.num_games,
        done as f64 / elapsed,
        stats.moves as f64 / elapsed,
        stats.avg_moves_per_game()
    );
}

fn play_game<G, Rep>(
    mcts: &mut Mcts<G, RepresentedEvaluator<G, Rep, BatcherClient>, NoExtraRules>,
    representation: &Rep,
    cfg: &SelfPlayConfig,
    should_stop: impl Fn() -> bool,
) -> Option<CompletedGame>
where
    G: GameState + Clone + Default,
    Rep: AlphaZeroRepresentation<G>,
{
    let mut game = G::default();
    let mut trajectory = Vec::with_capacity(cfg.max_moves.min(256));
    let mut rng = rand::rng();

    while !game.is_terminal() && trajectory.len() < cfg.max_moves {
        if should_stop() {
            return None;
        }
        let mut state = vec![0.0f32; Rep::state_size()];
        representation.encode_state(&game, &mut state);

        let result = mcts.search(&game, (), PolicyMode::Explore);
        let action = select_self_play_action(
            &result,
            mcts.config().variant,
            trajectory.len(),
            cfg.temperature_moves,
            &mut rng,
        );
        let policy = result
            .policy
            .iter()
            .map(|&(mv, probability)| (representation.move_to_action(&game, mv), probability))
            .collect();
        trajectory.push(Transition {
            state,
            policy,
            value: 0.0,
        });
        game.play(action);
    }

    let last_mover_value = last_mover_value(game.terminal_value(), game.is_terminal());
    assign_trajectory_values(&mut trajectory, last_mover_value);
    let stats = SelfPlayStats {
        games: 1,
        moves: trajectory.len(),
        ..Default::default()
    };
    Some(CompletedGame { stats, trajectory })
}

fn select_self_play_action<M: Copy, R: Rng + ?Sized>(
    result: &SearchResult<M>,
    variant: MctsVariant,
    ply: usize,
    temperature_moves: usize,
    rng: &mut R,
) -> M {
    if matches!(variant, MctsVariant::Gumbel { .. }) || ply >= temperature_moves {
        result.best_move()
    } else {
        result.sample_move(rng)
    }
}

/// Samples `p^(1/T)` without assigning mass to illegal/zero-probability
/// actions. `None` means deterministic argmax.
pub fn select_temperature_action<M: Copy, R: Rng + ?Sized>(
    result: &SearchResult<M>,
    temperature: Option<f32>,
    rng: &mut R,
) -> M {
    let Some(temperature) = temperature else {
        return result.best_move();
    };
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "self-play temperature must be finite and positive"
    );
    if temperature == 1.0 {
        return result.sample_move(rng);
    }

    let exponent = 1.0_f64 / f64::from(temperature);
    let log_probabilities: Vec<f64> = result
        .policy
        .iter()
        .map(|&(_, probability)| {
            if probability > 0.0 {
                f64::from(probability).ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let max_log_probability = log_probabilities
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_probabilities
        .into_iter()
        .map(|log_probability| ((log_probability - max_log_probability) * exponent).exp())
        .collect();
    let index = rand::distr::weighted::WeightedIndex::new(weights)
        .expect("search of a non-terminal position returns a non-empty policy")
        .sample(rng);
    result.policy[index].0
}

/// Walks the trajectory backwards from the terminal value, flipping sign
/// each ply so every transition's target value is from its own player's
/// perspective.
fn assign_trajectory_values(trajectory: &mut [Transition], terminal_value: f32) {
    let mut value = terminal_value;
    for t in trajectory.iter_mut().rev() {
        t.value = value;
        value = -value;
    }
}

fn last_mover_value(terminal_value: Option<TerminalValue>, terminal: bool) -> f32 {
    if terminal {
        -terminal_value.map_or(0.0, TerminalValue::as_f32)
    } else {
        TerminalValue::Draw.as_f32()
    }
}

#[cfg(test)]
mod tests;
