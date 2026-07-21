//! Chess-specific self-play orchestration.
//!
//! Search and replay storage remain generic algorithms. This module owns the
//! chess rules needed for repetition tracking, resignation, and the shared
//! evaluation cache.

use alphazero::representation::{
    AlphaZeroRepresentation, ChessAzRepresentation, ChessV1Representation,
};
use alphazero::ChessRepetitionRules;
use alphazero::{
    select_temperature_action, Batcher, BatcherClient, EvalTable, Mcts, MctsVariant, ReplayBuffer,
    RepresentedEvaluator, SelfPlayConfig, SelfPlayStats, Transition,
};
use engine_core::agent::PolicyMode;
use engine_core::{GameState, TerminalValue};
use games::{ChessGame, ChessPosition};
use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

struct CompletedGame {
    stats: SelfPlayStats,
    trajectory: Vec<Transition>,
}

/// Plays chess self-play while retaining full game history for repetition
/// rules and searching lightweight chess positions.
pub fn self_play_chess(
    batcher: &Batcher,
    replay: &ReplayBuffer,
    cfg: &SelfPlayConfig,
) -> SelfPlayStats {
    cfg.validate_chess_v2()
        .expect("invalid chess self-play configuration");
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
                let evaluator = RepresentedEvaluator::new(ChessV1Representation, batcher.client());
                let mut mcts = Mcts::new(evaluator, cfg.mcts, ChessRepetitionRules);
                if let Some(cache) = eval_cache {
                    mcts = mcts.with_eval_cache(cache);
                }
                while finished.load(Ordering::Relaxed) < cfg.num_games {
                    let Some(completed) = play_chess_game(&mut mcts, cfg, || {
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
                    add_stats(&mut stats.lock().unwrap(), completed.stats);
                    maybe_print_progress(done_before + 1, cfg, &stats, started);
                }
            });
        }
    });

    let mut stats = Arc::try_unwrap(stats).unwrap().into_inner().unwrap();
    if let Some(eval_cache) = eval_cache {
        let cache = eval_cache.stats();
        stats.tt_hits += cache.hits;
        stats.tt_misses += cache.misses;
        stats.tt_inserts += cache.inserts;
    }
    stats
}

/// Chess AZ v2 self-play using one authoritative game and a search snapshot.
pub fn self_play_chess_az_v2<const HISTORY: usize>(
    batcher: &Batcher,
    replay: &ReplayBuffer,
    cfg: &SelfPlayConfig,
) -> SelfPlayStats {
    cfg.validate_chess_v2()
        .expect("invalid chess AZ v2 self-play configuration");
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
                let evaluator =
                    RepresentedEvaluator::new(ChessAzRepresentation::<HISTORY>, batcher.client());
                let mut mcts = Mcts::new(evaluator, cfg.mcts, ChessRepetitionRules);
                if let Some(cache) = eval_cache {
                    mcts = mcts.with_eval_cache(cache);
                }
                while finished.load(Ordering::Relaxed) < cfg.num_games {
                    let Some(completed) = play_chess_az_v2_game::<HISTORY>(&mut mcts, cfg, || {
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
                    add_stats(&mut stats.lock().unwrap(), completed.stats);
                    maybe_print_progress(done_before + 1, cfg, &stats, started);
                }
            });
        }
    });

    let mut stats = Arc::try_unwrap(stats).unwrap().into_inner().unwrap();
    if let Some(eval_cache) = eval_cache {
        let cache = eval_cache.stats();
        stats.tt_hits += cache.hits;
        stats.tt_misses += cache.misses;
        stats.tt_inserts += cache.inserts;
    }
    stats
}

fn play_chess_game(
    mcts: &mut Mcts<
        ChessPosition,
        RepresentedEvaluator<ChessPosition, ChessV1Representation, BatcherClient>,
        ChessRepetitionRules,
    >,
    cfg: &SelfPlayConfig,
    should_stop: impl Fn() -> bool,
) -> Option<CompletedGame> {
    let mut game = ChessGame::default();
    let mut trajectory = Vec::with_capacity(cfg.max_moves.min(256));
    let mut rng = rand::rng();
    let resignation_disabled =
        rng.random_bool(cfg.resignation_disable_probability.clamp(0.0, 1.0) as f64);
    let mut resignation_streak = 0;
    let mut resigned_value = None;
    let mut stats = SelfPlayStats {
        games: 1,
        ..Default::default()
    };

    while !game.is_terminal() && trajectory.len() < cfg.max_moves {
        if should_stop() {
            return None;
        }
        let search_state = game.position();
        let mut state = vec![
            0.0;
            <ChessV1Representation as AlphaZeroRepresentation<ChessPosition>>::state_size()
        ];
        ChessV1Representation.encode_state(&search_state, &mut state);

        let full_search = rng.random_bool(cfg.full_simulation_probability.clamp(0.0, 1.0) as f64);
        if let Some(profiles) = cfg.chess_v2_gumbel_profiles {
            let profile = if full_search {
                stats.full_searches += 1;
                profiles.full
            }
            else {
                stats.fast_searches += 1;
                profiles.fast
            };
            mcts.set_gumbel_profile(profile);
        }
        else {
            let simulations = if full_search {
                stats.full_searches += 1;
                cfg.mcts.simulations
            }
            else {
                stats.fast_searches += 1;
                cfg.fast_simulations.max(1)
            };
            mcts.set_simulations(simulations);
        }

        let result = mcts.search(
            &search_state,
            game.repetition_context(),
            PolicyMode::Explore,
        );
        let action = if let Some(temperature) = cfg.chess_v2_temperature {
            select_temperature_action(&result, temperature.at_ply(trajectory.len()), &mut rng)
        }
        else if matches!(mcts.config().variant, MctsVariant::Gumbel { .. })
            || trajectory.len() >= cfg.temperature_moves
        {
            result.best_move()
        }
        else {
            result.sample_move(&mut rng)
        };
        let search_value = result.value;
        let policy = result
            .policy
            .iter()
            .map(|&(mv, probability)| {
                (
                    ChessV1Representation.move_to_action(&search_state, mv),
                    probability,
                )
            })
            .collect();
        trajectory.push(Transition {
            state,
            policy,
            value: 0.0,
        });

        if cfg.resignation_enabled
            && !resignation_disabled
            && trajectory.len() >= cfg.resignation_min_ply
            && search_value < cfg.resignation_threshold
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
        game.play(action);
    }

    let mut value =
        resigned_value.unwrap_or_else(|| -game.terminal_value().map_or(0.0, TerminalValue::as_f32));
    for transition in trajectory.iter_mut().rev() {
        transition.value = value;
        value = -value;
    }
    stats.moves = trajectory.len();
    Some(CompletedGame { stats, trajectory })
}

fn play_chess_az_v2_game<const HISTORY: usize>(
    mcts: &mut Mcts<
        alphazero::representation::ChessAzState<HISTORY>,
        RepresentedEvaluator<
            alphazero::representation::ChessAzState<HISTORY>,
            ChessAzRepresentation<HISTORY>,
            BatcherClient,
        >,
        ChessRepetitionRules,
    >,
    cfg: &SelfPlayConfig,
    should_stop: impl Fn() -> bool,
) -> Option<CompletedGame> {
    let mut game = ChessGame::default();
    let mut trajectory = Vec::with_capacity(cfg.max_moves.min(256));
    let mut rng = rand::rng();
    let resignation_disabled =
        rng.random_bool(cfg.resignation_disable_probability.clamp(0.0, 1.0) as f64);
    let mut resignation_streak = 0;
    let mut resigned_value = None;
    let mut stats = SelfPlayStats {
        games: 1,
        ..Default::default()
    };

    while !game.is_terminal() && trajectory.len() < cfg.max_moves {
        if should_stop() {
            return None;
        }
        let search_state = alphazero::representation::ChessAzState::from_game(&game);
        let mut encoded = vec![
            0.0;
            <ChessAzRepresentation<HISTORY> as AlphaZeroRepresentation<
                alphazero::representation::ChessAzState<HISTORY>,
            >>::state_size()
        ];
        ChessAzRepresentation::<HISTORY>.encode_state(&search_state, &mut encoded);

        let full_search = rng.random_bool(cfg.full_simulation_probability as f64);
        if let Some(profiles) = cfg.chess_v2_gumbel_profiles {
            let profile = if full_search {
                stats.full_searches += 1;
                profiles.full
            }
            else {
                stats.fast_searches += 1;
                profiles.fast
            };
            mcts.set_gumbel_profile(profile);
        }
        else {
            let simulations = if full_search {
                stats.full_searches += 1;
                cfg.mcts.simulations
            }
            else {
                stats.fast_searches += 1;
                cfg.fast_simulations.max(1)
            };
            mcts.set_simulations(simulations);
        }

        let repetition_context = game.repetition_context();
        let result = mcts.search(&search_state, repetition_context, PolicyMode::Explore);
        let action = select_temperature_action(
            &result,
            cfg.chess_v2_temperature
                .expect("chess AZ v2 requires a temperature schedule")
                .at_ply(trajectory.len()),
            &mut rng,
        );
        let search_value = result.value;
        let policy = result
            .policy
            .iter()
            .map(|&(mv, probability)| {
                (
                    ChessAzRepresentation::<HISTORY>.move_to_action(&search_state, mv),
                    probability,
                )
            })
            .collect();
        trajectory.push(Transition {
            state: encoded,
            policy,
            value: 0.0,
        });

        if cfg.resignation_enabled
            && !resignation_disabled
            && trajectory.len() >= cfg.resignation_min_ply
            && search_value < cfg.resignation_threshold
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
        game.play(action);
    }

    let mut value =
        resigned_value.unwrap_or_else(|| -game.terminal_value().map_or(0.0, TerminalValue::as_f32));
    for transition in trajectory.iter_mut().rev() {
        transition.value = value;
        value = -value;
    }
    stats.moves = trajectory.len();
    Some(CompletedGame { stats, trajectory })
}

fn claim_completed_game(finished: &AtomicUsize, target: usize) -> Result<usize, usize> {
    finished.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |done| {
        (done < target).then_some(done + 1)
    })
}

fn add_stats(total: &mut SelfPlayStats, game: SelfPlayStats) {
    total.games += game.games;
    total.moves += game.moves;
    total.full_searches += game.full_searches;
    total.fast_searches += game.fast_searches;
    total.resignations += game.resignations;
}

fn maybe_print_progress(
    done: usize,
    cfg: &SelfPlayConfig,
    stats: &Mutex<SelfPlayStats>,
    started: Instant,
) {
    if cfg.progress_every == 0
        || (done != cfg.num_games && !done.is_multiple_of(cfg.progress_every))
    {
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
