//! The owned, reloadable AlphaZero training lifecycle.

use crate::experiment::{ExperimentConfig, InferenceEngine, RunDir, RunState};
use crate::inference::{InferenceService, InferenceSource};
use crate::representation::{
    ChessAzRepresentation, ChessClassicRepresentation, Connect4AzRepresentation,
};
use crate::selfplay::{
    ChessCanonicalSelfPlayDomain, ChessClassicSelfPlayDomain, DomainSelfPlayWorkerFactory,
    SelfPlayCoordinator, SelfPlayEpoch, SelfPlayStats, SelfPlayWorkerFactory,
    StandardSelfPlayDomain,
};
use crate::{
    AlphaZeroRepresentation, BatcherStats, ChessHistory, GameKind, Network, ReplayBuffer,
    TrainMetrics, Trainer, TrainingSeed,
};
use anyhow::Result;
use engine_core::GameState;
use games::{ChessPosition, Connect4};
use serde::Serialize;
use std::path::Path;
use tch::{nn, Device};

/// The amount of training work requested by [`TrainingRun::run`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunLimit {
    Iterations(usize),
    Forever,
}

/// The durable and operational results of one full training iteration.
#[derive(Debug, Serialize)]
pub struct IterationReport {
    pub iteration: u64,
    pub model_generation: u64,
    pub total_games_generated: u64,
    pub games: usize,
    pub moves: usize,
    pub replay_samples: usize,
    pub training: Option<TrainMetrics>,
    pub inference: BatcherStats,
}

/// Runtime-dispatched training run. Its inner variants keep search and replay
/// code statically dispatched for each representation.
pub struct TrainingRun {
    inner: TrainingRunKind,
}

enum TrainingRunKind {
    Connect4(Connect4TrainingRun),
    ChessClassic(ChessClassicTrainingRun),
    ChessH1(ChessCanonicalTrainingRun<1>),
    ChessH4(ChessCanonicalTrainingRun<4>),
    ChessH8(ChessCanonicalTrainingRun<8>),
}

type Connect4Workers =
    DomainSelfPlayWorkerFactory<StandardSelfPlayDomain<Connect4, Connect4AzRepresentation>>;
type Connect4TrainingRun = TypedTrainingRun<Connect4, Connect4AzRepresentation, Connect4Workers>;

type ChessClassicWorkers = DomainSelfPlayWorkerFactory<ChessClassicSelfPlayDomain>;
type ChessClassicTrainingRun =
    TypedTrainingRun<ChessPosition, ChessClassicRepresentation, ChessClassicWorkers>;

type ChessCanonicalWorkers<const HISTORY: usize> =
    DomainSelfPlayWorkerFactory<ChessCanonicalSelfPlayDomain<HISTORY>>;
type ChessCanonicalTrainingRun<const HISTORY: usize> = TypedTrainingRun<
    crate::representation::ChessAzState<HISTORY>,
    ChessAzRepresentation<HISTORY>,
    ChessCanonicalWorkers<HISTORY>,
>;

struct TypedTrainingRun<S, R, F>
where
    S: GameState + Clone + Send + Sync + 'static,
    S::Move: Send + Sync,
    R: AlphaZeroRepresentation<S>,
    F: SelfPlayWorkerFactory<S>,
{
    run_dir: RunDir,
    experiment: ExperimentConfig,
    state: RunState,
    device: Device,
    var_store: nn::VarStore,
    network: Network,
    inference: InferenceService,
    replay: ReplayBuffer<S>,
    self_play: SelfPlayCoordinator,
    workers: F,
    trainer: Trainer,
    representation: R,
}

struct TypedRunComponents<S, F> {
    run_dir: RunDir,
    experiment: ExperimentConfig,
    state: RunState,
    device: Device,
    var_store: nn::VarStore,
    network: Network,
    inference: InferenceService,
    replay: ReplayBuffer<S>,
    workers: F,
}

impl TrainingRun {
    /// Opens an initialized run with reloadable native inference.
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let (run_dir, experiment, mut state) = RunDir::open(path)?;
        anyhow::ensure!(
            experiment.inference.engine == InferenceEngine::Native,
            "TrainingRun requires reloadable native inference; TensorRT TorchScript supports one-generation self-play only"
        );

        let (var_store, network) = load_training_model(&run_dir, &experiment, &mut state, device)?;
        let inference = InferenceService::load(
            &experiment.model,
            InferenceSource::Checkpoint(&run_dir.latest_path()),
            device,
            &experiment.inference,
        )?;

        let inner = match experiment.model.game() {
            GameKind::Connect4 => TrainingRunKind::Connect4(open_connect4(
                run_dir, experiment, state, device, var_store, network, inference,
            )?),
            GameKind::Chess => open_chess(
                run_dir, experiment, state, device, var_store, network, inference,
            )?,
        };

        Ok(Self { inner })
    }

    pub fn step(&mut self) -> Result<IterationReport> {
        match &mut self.inner {
            TrainingRunKind::Connect4(run) => run.step(),
            TrainingRunKind::ChessClassic(run) => run.step(),
            TrainingRunKind::ChessH1(run) => run.step(),
            TrainingRunKind::ChessH4(run) => run.step(),
            TrainingRunKind::ChessH8(run) => run.step(),
        }
    }

    pub fn run(&mut self, limit: RunLimit) -> Result<()> {
        match limit {
            RunLimit::Iterations(iterations) => {
                for _ in 0..iterations {
                    self.step()?;
                }
            }
            RunLimit::Forever => loop {
                self.step()?;
            },
        }
        Ok(())
    }

    pub fn state(&self) -> &RunState {
        match &self.inner {
            TrainingRunKind::Connect4(run) => &run.state,
            TrainingRunKind::ChessClassic(run) => &run.state,
            TrainingRunKind::ChessH1(run) => &run.state,
            TrainingRunKind::ChessH4(run) => &run.state,
            TrainingRunKind::ChessH8(run) => &run.state,
        }
    }
}

impl<S, R, F> TypedTrainingRun<S, R, F>
where
    S: GameState + Clone + Send + Sync + 'static,
    S::Move: Send + Sync,
    R: AlphaZeroRepresentation<S>,
    F: SelfPlayWorkerFactory<S>,
{
    fn step(&mut self) -> Result<IterationReport> {
        let self_play = self.generate_self_play()?;
        let training = self.train_network();

        let next_state = self.advance_state(&self_play, training.as_ref());
        let next_state = self.save_checkpoint(next_state)?;
        self.reload_inference()?;

        self.state = next_state;
        let report = self.build_report(self_play, training);
        self.persist_report(&report)?;

        Ok(report)
    }

    fn generate_self_play(&self) -> Result<SelfPlayStats> {
        let epoch = SelfPlayEpoch {
            model_generation: self.state.model_generation,
            first_game_id: self.state.total_games_generated,
        };

        self.self_play.run(&self.workers, &self.replay, epoch)
    }

    fn train_network(&mut self) -> Option<TrainMetrics> {
        self.trainer.train(
            &self.network,
            &self.replay,
            &self.representation,
            self.device,
            TrainingSeed {
                experiment_seed: self.experiment.seed,
                global_step: self.state.global_step,
            },
        )
    }

    fn advance_state(
        &self,
        self_play: &SelfPlayStats,
        training: Option<&TrainMetrics>,
    ) -> RunState {
        let mut state = self.state.clone();
        state.total_games_generated += self_play.games as u64;
        state.iteration += 1;
        state.model_generation += 1;
        state.global_step += training.map_or(0, |metrics| metrics.train_steps as u64);
        state.replay_sample_count = self.replay.len();
        state.optimizer_moments_restored = false;

        state
    }

    fn save_checkpoint(&self, mut state: RunState) -> Result<RunState> {
        self.run_dir
            .write_latest(&mut state, |path| Ok(self.var_store.save(path)?))?;

        Ok(state)
    }

    fn reload_inference(&self) -> Result<()> {
        self.inference
            .reload_weights(&self.run_dir.latest_path())
            .map_err(anyhow::Error::msg)
    }

    fn build_report(
        &self,
        self_play: SelfPlayStats,
        training: Option<TrainMetrics>,
    ) -> IterationReport {
        IterationReport {
            iteration: self.state.iteration,
            model_generation: self.state.model_generation,
            total_games_generated: self.state.total_games_generated,
            games: self_play.games,
            moves: self_play.moves,
            replay_samples: self.replay.len(),
            training,
            inference: self.inference.stats(),
        }
    }

    fn persist_report(&self, report: &IterationReport) -> Result<()> {
        self.run_dir.log_metrics(serde_json::json!({
            "iteration": report.iteration,
            "model_generation": report.model_generation,
            "total_games_generated": report.total_games_generated,
            "games": report.games,
            "moves": report.moves,
            "replay_samples": report.replay_samples,
            "policy_loss": report.training.as_ref().map(|metrics| metrics.policy_loss),
            "value_loss": report.training.as_ref().map(|metrics| metrics.value_loss),
            "learning_rate": report.training.as_ref().map(|metrics| metrics.learning_rate),
            "training": &report.training,
            "inference": report.inference,
        }))
    }
}

fn load_training_model(
    run_dir: &RunDir,
    experiment: &ExperimentConfig,
    state: &mut RunState,
    device: Device,
) -> Result<(nn::VarStore, Network)> {
    let mut var_store = nn::VarStore::new(device);
    let network = Network::new(&var_store.root(), &experiment.model)?;

    if let Some(checkpoint) = run_dir.latest_checkpoint(state) {
        var_store.load(checkpoint)?;
        state.mark_weights_only_resume();
        run_dir.write_state(state)?;
        log_weights_only_resume(run_dir)?;
    }
    else {
        run_dir.write_latest(state, |path| Ok(var_store.save(path)?))?;
    }

    Ok((var_store, network))
}

fn log_weights_only_resume(run_dir: &RunDir) -> Result<()> {
    eprintln!("WARNING: resumed checkpoint weights only; replay and optimizer moments were reset.");

    run_dir.log_metrics(serde_json::json!({
        "event": "resume",
        "resume_kind": "weights-only",
        "replay_restored": false,
        "optimizer_moments_restored": false,
    }))
}

fn open_connect4(
    run_dir: RunDir,
    experiment: ExperimentConfig,
    state: RunState,
    device: Device,
    var_store: nn::VarStore,
    network: Network,
    inference: InferenceService,
) -> Result<Connect4TrainingRun> {
    let replay = ReplayBuffer::new(experiment.replay.capacity, experiment.model.action_size());
    let workers = DomainSelfPlayWorkerFactory::new(&inference, experiment.self_play.clone())?;

    TypedTrainingRun::assemble(
        TypedRunComponents {
            run_dir,
            experiment,
            state,
            device,
            var_store,
            network,
            inference,
            replay,
            workers,
        },
        Connect4AzRepresentation,
    )
}

fn open_chess(
    run_dir: RunDir,
    experiment: ExperimentConfig,
    state: RunState,
    device: Device,
    var_store: nn::VarStore,
    network: Network,
    inference: InferenceService,
) -> Result<TrainingRunKind> {
    if experiment.model.is_chess_classic() {
        return Ok(TrainingRunKind::ChessClassic(open_chess_classic(
            run_dir, experiment, state, device, var_store, network, inference,
        )?));
    }

    match experiment
        .model
        .chess_history()
        .expect("validated chess model")
    {
        ChessHistory::One => Ok(TrainingRunKind::ChessH1(open_chess_history::<1>(
            run_dir, experiment, state, device, var_store, network, inference,
        )?)),
        ChessHistory::Four => Ok(TrainingRunKind::ChessH4(open_chess_history::<4>(
            run_dir, experiment, state, device, var_store, network, inference,
        )?)),
        ChessHistory::Eight => Ok(TrainingRunKind::ChessH8(open_chess_history::<8>(
            run_dir, experiment, state, device, var_store, network, inference,
        )?)),
    }
}

fn open_chess_classic(
    run_dir: RunDir,
    experiment: ExperimentConfig,
    state: RunState,
    device: Device,
    var_store: nn::VarStore,
    network: Network,
    inference: InferenceService,
) -> Result<ChessClassicTrainingRun> {
    let replay = ReplayBuffer::new(experiment.replay.capacity, experiment.model.action_size());
    let workers = DomainSelfPlayWorkerFactory::new(&inference, experiment.self_play.clone())?;

    TypedTrainingRun::assemble(
        TypedRunComponents {
            run_dir,
            experiment,
            state,
            device,
            var_store,
            network,
            inference,
            replay,
            workers,
        },
        ChessClassicRepresentation,
    )
}

fn open_chess_history<const HISTORY: usize>(
    run_dir: RunDir,
    experiment: ExperimentConfig,
    state: RunState,
    device: Device,
    var_store: nn::VarStore,
    network: Network,
    inference: InferenceService,
) -> Result<ChessCanonicalTrainingRun<HISTORY>> {
    let replay = ReplayBuffer::new(experiment.replay.capacity, experiment.model.action_size());
    let workers = DomainSelfPlayWorkerFactory::new(&inference, experiment.self_play.clone())?;

    TypedTrainingRun::assemble(
        TypedRunComponents {
            run_dir,
            experiment,
            state,
            device,
            var_store,
            network,
            inference,
            replay,
            workers,
        },
        ChessAzRepresentation,
    )
}

impl<S, R, F> TypedTrainingRun<S, R, F>
where
    S: GameState + Clone + Send + Sync + 'static,
    S::Move: Send + Sync,
    R: AlphaZeroRepresentation<S>,
    F: SelfPlayWorkerFactory<S>,
{
    fn assemble(components: TypedRunComponents<S, F>, representation: R) -> Result<Self> {
        let TypedRunComponents {
            run_dir,
            experiment,
            state,
            device,
            var_store,
            network,
            inference,
            replay,
            workers,
        } = components;
        let self_play = SelfPlayCoordinator::new(experiment.self_play.clone(), experiment.seed)?;
        let trainer = Trainer::new(&var_store, experiment.training.clone())?;

        Ok(Self {
            run_dir,
            experiment,
            state,
            device,
            var_store,
            network,
            inference,
            replay,
            self_play,
            workers,
            trainer,
            representation,
        })
    }
}
