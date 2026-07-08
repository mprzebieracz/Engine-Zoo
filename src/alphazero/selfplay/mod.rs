use super::batcher::{Batcher, BatcherClient};
use super::mcts::{EvalTable, EvalTableStats, Mcts, MctsConfig};
use super::replay::{ReplayBuffer, Transition};
use crate::game::{Action, Game};
use crate::games::chess::ChessGame;
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
        }
        else {
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

    fn add_tt(&mut self, tt: EvalTableStats) {
        self.tt_hits += tt.hits;
        self.tt_misses += tt.misses;
        self.tt_inserts += tt.inserts;
    }
}

#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    pub num_games: usize,
    pub threads: usize,
    /// Games longer than this are truncated and scored as draws.
    pub max_moves: usize,
    /// Moves sampled from the visit distribution before switching to argmax.
    pub temperature_moves: usize,
    /// Print self-play progress every N completed games. Set 0 to disable.
    pub progress_every: usize,
    /// Maximum chess NN evaluation cache entries for one self-play iteration.
    /// Set to 0 to disable the cache.
    pub tt_entries: usize,
    pub fast_simulations: usize,
    pub full_simulation_probability: f32,
    pub resignation_enabled: bool,
    pub resignation_threshold: f32,
    pub resignation_consecutive_moves: usize,
    pub resignation_min_ply: usize,
    pub resignation_disable_probability: f32,
    pub mcts: MctsConfig,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            num_games: 100,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            max_moves: 256,
            temperature_moves: 30,
            progress_every: 25,
            tt_entries: 1_000_000,
            fast_simulations: 100,
            full_simulation_probability: 0.25,
            resignation_enabled: true,
            resignation_threshold: -0.95,
            resignation_consecutive_moves: 3,
            resignation_min_ply: 60,
            resignation_disable_probability: 0.10,
            mcts: MctsConfig::default(),
        }
    }
}

/// Plays `cfg.num_games` games of self-play across `cfg.threads` threads, all
/// sharing `batcher` for network evaluation, appending trajectories to `replay`.
pub fn self_play<G: Game>(
    batcher: &Batcher,
    replay: &ReplayBuffer,
    cfg: &SelfPlayConfig,
) -> SelfPlayStats {
    let finished = AtomicUsize::new(0);
    let started = Instant::now();
    let stats = Arc::new(Mutex::new(SelfPlayStats::default()));

    std::thread::scope(|scope| {
        for _ in 0..cfg.threads.max(1) {
            let stats = Arc::clone(&stats);
            let finished = &finished;
            scope.spawn(move || {
                let mut mcts = Mcts::new(batcher.client(), cfg.mcts);
                while finished.load(Ordering::Relaxed) < cfg.num_games {
                    let Some(completed) = play_game::<G>(&mut mcts, replay, cfg, || {
                        finished.load(Ordering::Relaxed) >= cfg.num_games
                    })
                    else {
                        break;
                    };
                    let Ok(done_before) = claim_completed_game(finished, cfg.num_games)
                    else {
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

/// Chess-specific self-play path: MCTS searches on the lightweight
/// `ChessPosition` while the full `ChessGame` owns real-game repetition
/// history for replay generation.
pub fn self_play_chess(
    batcher: &Batcher,
    replay: &ReplayBuffer,
    cfg: &SelfPlayConfig,
) -> SelfPlayStats {
    let finished = AtomicUsize::new(0);
    let started = Instant::now();
    let eval_cache = (cfg.tt_entries > 0).then(|| Arc::new(EvalTable::new(cfg.tt_entries)));
    let stats = Arc::new(Mutex::new(SelfPlayStats::default()));

    std::thread::scope(|scope| {
        for _ in 0..cfg.threads.max(1) {
            let eval_cache = eval_cache.clone();
            let stats = Arc::clone(&stats);
            let finished = &finished;
            scope.spawn(move || {
                let mut mcts = Mcts::new(batcher.client(), cfg.mcts);
                if let Some(cache) = eval_cache {
                    mcts = mcts.with_eval_cache(cache);
                }
                while finished.load(Ordering::Relaxed) < cfg.num_games {
                    let Some(completed) = play_chess_game(&mut mcts, replay, cfg, || {
                        finished.load(Ordering::Relaxed) >= cfg.num_games
                    })
                    else {
                        break;
                    };
                    let Ok(done_before) = claim_completed_game(finished, cfg.num_games)
                    else {
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
    let mut stats = Arc::try_unwrap(stats).unwrap().into_inner().unwrap();
    if let Some(eval_cache) = eval_cache {
        stats.add_tt(eval_cache.stats());
    }
    stats
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
        "self-play progress: {done}/{} games, {:.1} games/s, {:.1} moves/game",
        cfg.num_games,
        done as f64 / elapsed,
        stats.avg_moves_per_game()
    );
}

fn play_game<G: Game>(
    mcts: &mut Mcts<BatcherClient>,
    _replay: &ReplayBuffer,
    cfg: &SelfPlayConfig,
    should_stop: impl Fn() -> bool,
) -> Option<CompletedGame> {
    let mut game = G::default();
    let mut trajectory: Vec<Transition> = Vec::new();
    let mut rng = rand::rng();

    while !game.is_terminal() && trajectory.len() < cfg.max_moves {
        if should_stop() {
            return None;
        }
        let mut state = vec![0.0f32; G::state_size()];
        game.encode_state(&mut state);

        let result = mcts.search(&game);
        let action = if trajectory.len() < cfg.temperature_moves {
            result.sample_action(&mut rng)
        }
        else {
            result.best_action()
        };

        let policy = result
            .policy
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(a, &p)| (a as Action, p))
            .collect();

        trajectory.push(Transition {
            state,
            policy,
            reward: 0.0,
        });
        game.step(action);
    }

    assign_trajectory_rewards(&mut trajectory, -game.reward());
    let stats = SelfPlayStats {
        games: 1,
        moves: trajectory.len(),
        ..Default::default()
    };
    Some(CompletedGame { stats, trajectory })
}

fn play_chess_game(
    mcts: &mut Mcts<BatcherClient>,
    _replay: &ReplayBuffer,
    cfg: &SelfPlayConfig,
    should_stop: impl Fn() -> bool,
) -> Option<CompletedGame> {
    let mut game = ChessGame::default();
    let mut trajectory: Vec<Transition> = Vec::new();
    let mut rng = rand::rng();
    let resignation_disabled =
        rng.random_bool(cfg.resignation_disable_probability.clamp(0.0, 1.0) as f64);
    let mut resignation_streak = 0usize;
    let mut resigned_value = None;
    let mut stats = SelfPlayStats {
        games: 1,
        ..Default::default()
    };

    while !game.is_terminal() && trajectory.len() < cfg.max_moves {
        if should_stop() {
            return None;
        }
        let mut state = vec![0.0f32; ChessGame::state_size()];
        game.encode_state(&mut state);

        let full_search = rng.random_bool(cfg.full_simulation_probability.clamp(0.0, 1.0) as f64);
        let simulations = if full_search {
            stats.full_searches += 1;
            cfg.mcts.simulations
        }
        else {
            stats.fast_searches += 1;
            cfg.fast_simulations.max(1)
        };
        mcts.set_simulations(simulations);

        let position = game.position();
        let result =
            mcts.search_with_repetitions(&position, |hash| game.repetitions_before_current(hash));
        let action = if trajectory.len() < cfg.temperature_moves {
            result.sample_action(&mut rng)
        }
        else {
            result.best_action()
        };

        let policy = result
            .policy
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(a, &p)| (a as Action, p))
            .collect();

        trajectory.push(Transition {
            state,
            policy,
            reward: 0.0,
        });

        if cfg.resignation_enabled
            && !resignation_disabled
            && trajectory.len() >= cfg.resignation_min_ply
            && result.value < cfg.resignation_threshold
        {
            resignation_streak += 1;
        }
        else {
            resignation_streak = 0;
        }
        if resignation_streak >= cfg.resignation_consecutive_moves.max(1) {
            resigned_value = Some(-1.0);
            stats.resignations += 1;
            break;
        }

        game.step(action);
    }

    let terminal_reward = resigned_value.unwrap_or_else(|| -game.reward());
    assign_trajectory_rewards(&mut trajectory, terminal_reward);
    stats.moves = trajectory.len();
    Some(CompletedGame { stats, trajectory })
}

/// Walks the trajectory backwards from the terminal reward, flipping sign
/// each ply so every transition's target value is from its own player's
/// perspective.
fn assign_trajectory_rewards(trajectory: &mut [Transition], terminal_reward: f32) {
    let mut value = terminal_reward;
    for t in trajectory.iter_mut().rev() {
        t.reward = value;
        value = -value;
    }
}

#[cfg(test)]
mod tests;
