use crate::{ReplayBuffer, ReplaySample};
use anyhow::{ensure, Result};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Instant;

use super::SelfPlayConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameRequest {
    pub game_id: u64,
    pub model_generation: u64,
    pub worker_id: usize,
    pub experiment_seed: u64,
}

impl GameRequest {
    pub fn seed_for(self, purpose: u64) -> u64 {
        splitmix64(
            self.experiment_seed
                ^ splitmix64(self.game_id)
                ^ splitmix64(self.model_generation)
                ^ purpose,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelfPlayEpoch {
    pub model_generation: u64,
    pub first_game_id: u64,
}

pub(crate) const BUDGET_SEED: u64 = 0x62_75_64_67_65_74;
pub(crate) const MOVE_SEED: u64 = 0x6d_6f_76_65;
pub(crate) const RESIGNATION_SEED: u64 = 0x72_65_73_69_67_6e;
pub(crate) const SEARCH_SEED: u64 = 0x73_65_61_72_63_68;

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub trait SelfPlayWorker<S>: Send {
    fn play_game(&mut self, game: GameRequest) -> Result<CompletedGame<S>>;
}

pub trait SelfPlayWorkerFactory<S>: Sync {
    type Worker: SelfPlayWorker<S>;

    fn create(&self, worker_id: usize, seed: u64) -> Result<Self::Worker>;
}

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

impl SelfPlayStats {
    pub fn avg_moves_per_game(&self) -> f64 {
        self.moves as f64 / self.games.max(1) as f64
    }

    pub fn add(&mut self, other: Self) {
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

pub struct CompletedGame<S> {
    pub stats: SelfPlayStats,
    pub trajectory: Vec<ReplaySample<S>>,
}

pub struct SelfPlayCoordinator {
    config: SelfPlayConfig,
    seed: u64,
}

impl SelfPlayCoordinator {
    pub fn new(config: SelfPlayConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, seed })
    }

    pub fn run<S: Send + Sync, F: SelfPlayWorkerFactory<S>>(
        &self,
        factory: &F,
        replay: &ReplayBuffer<S>,
        epoch: SelfPlayEpoch,
    ) -> Result<SelfPlayStats> {
        ensure!(
            u64::try_from(self.config.num_games)
                .ok()
                .and_then(|games| epoch.first_game_id.checked_add(games))
                .is_some(),
            "self-play game IDs overflow u64"
        );
        let next_game = AtomicUsize::new(0);
        let cancelled = AtomicBool::new(false);
        let failure = Mutex::new(None);
        let started = Instant::now();
        let mut stats = SelfPlayStats::default();

        std::thread::scope(|scope| {
            let (sender, receiver) = mpsc::channel();
            for worker_id in 0..self.config.threads {
                let mut worker = match factory.create(worker_id, worker_seed(self.seed, worker_id))
                {
                    Ok(worker) => worker,
                    Err(error) => {
                        *failure.lock().unwrap() = Some(error.context("creating self-play worker"));
                        cancelled.store(true, Ordering::Release);
                        break;
                    }
                };
                let next_game = &next_game;
                let cancelled = &cancelled;
                let failure = &failure;
                let sender = sender.clone();
                scope.spawn(move || loop {
                    if cancelled.load(Ordering::Acquire) {
                        break;
                    }
                    let game_id = next_game.fetch_add(1, Ordering::Relaxed);
                    if game_id >= self.config.num_games {
                        break;
                    }
                    let request = GameRequest {
                        game_id: epoch.first_game_id + game_id as u64,
                        model_generation: epoch.model_generation,
                        worker_id,
                        experiment_seed: self.seed,
                    };
                    match worker.play_game(request) {
                        Ok(completed) => {
                            if cancelled.load(Ordering::Acquire) {
                                break;
                            }
                            if sender.send((request.game_id, completed)).is_err() {
                                break;
                            }
                        }
                        Err(error) => {
                            cancelled.store(true, Ordering::Release);
                            let mut slot = failure.lock().unwrap();
                            if slot.is_none() {
                                *slot = Some(error.context(format!("self-play game {game_id}")));
                            }
                            break;
                        }
                    }
                });
            }
            drop(sender);

            let mut pending = BTreeMap::new();
            let mut next_commit = epoch.first_game_id;
            while let Ok((game_id, completed)) = receiver.recv() {
                pending.insert(game_id, completed);
                while let Some(completed) = pending.remove(&next_commit) {
                    replay.add(completed.trajectory);
                    stats.add(completed.stats);
                    print_progress(stats.games, &self.config, &stats, started);
                    next_commit += 1;
                }
            }
        });
        if let Some(error) = failure.into_inner().unwrap() {
            return Err(error);
        }
        Ok(stats)
    }
}

fn worker_seed(experiment_seed: u64, worker_id: usize) -> u64 {
    splitmix64(experiment_seed ^ worker_id as u64)
}

fn print_progress(done: usize, config: &SelfPlayConfig, stats: &SelfPlayStats, started: Instant) {
    if config.progress_every == 0 || (done != config.num_games && done % config.progress_every != 0)
    {
        return;
    }
    let elapsed = started.elapsed().as_secs_f64().max(1e-6);
    println!(
        "self-play progress: {done}/{} games, {:.1} games/s, {:.1} positions/s, {:.1} moves/game",
        config.num_games,
        done as f64 / elapsed,
        stats.moves as f64 / elapsed,
        stats.avg_moves_per_game(),
    );
}
