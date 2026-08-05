//! The owned AlphaZero training lifecycle.

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
    ReplaySampler, SelfPlayProgress, TrainMetrics, TrainProgress, Trainer, TrainingInvocation,
    TrainingSeed,
};
use anyhow::{Context, Result};
use engine_core::GameState;
use games::{ChessPosition, Connect4};
use serde::Serialize;
use std::path::{Path, PathBuf};
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
    /// What the caller must do before self-play can use the next checkpoint.
    pub next_inference: NextInference,
}

/// The inference-model action required after an iteration.
///
/// Native inference reloads the checkpoint in-place. A TensorRT TorchScript
/// module has fixed weights, so the application must compile the checkpoint
/// named here, replace the configured module, and reload TensorRT before the
/// following self-play generation.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum NextInference {
    NativeReloaded,
    TensorRtRecompileRequired {
        checkpoint: PathBuf,
        compiled_artifact: PathBuf,
    },
}

#[derive(Clone, Copy)]
struct IterationFacts {
    games: usize,
    trained_steps: usize,
    replay_samples: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum InferenceRefreshPlan {
    ReloadNative {
        checkpoint: PathBuf,
    },
    RequireTensorRtRecompile {
        checkpoint: PathBuf,
        compiled_artifact: PathBuf,
    },
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
    make_workers: fn(&InferenceService, crate::SelfPlayConfig) -> Result<F>,
    trainer: Trainer,
    sampler: ReplaySampler<R>,
    pending_tensor_rt_recompile: Option<NextInference>,
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
    make_workers: fn(&InferenceService, crate::SelfPlayConfig) -> Result<F>,
}

impl TrainingRun {
    /// Opens an initialized run using the experiment stored on disk.
    ///
    /// TensorRT uses a fixed module. After each checkpoint, call
    /// [`Self::reload_tensor_rt_inference`] once the configured module has
    /// been rebuilt, before requesting another generation.
    pub fn open(path: &Path, device: Device) -> Result<Self> {
        let (run_dir, experiment, state) = RunDir::open_writer(path)?;
        Self::from_opened(run_dir, experiment, state, device)
    }

    /// Opens an initialized run with a caller-supplied experiment override.
    ///
    /// Used by runtime flags such as `train run --cache`, which select the raw
    /// TensorRT backend without rewriting the immutable `experiment.toml`.
    pub fn open_with_experiment(
        path: &Path,
        experiment: ExperimentConfig,
        device: Device,
    ) -> Result<Self> {
        experiment.validate()?;
        let (run_dir, _, state) = RunDir::open_writer(path)?;
        Self::from_opened(run_dir, experiment, state, device)
    }

    fn from_opened(
        run_dir: RunDir,
        experiment: ExperimentConfig,
        mut state: RunState,
        device: Device,
    ) -> Result<Self> {
        let (var_store, network) = load_training_model(&run_dir, &experiment, &mut state, device)?;
        let inference = load_inference_service(&run_dir, &experiment, &state, device)?;

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
        self.step_with_progress(|_| {})
    }

    pub fn step_with_progress<F>(&mut self, progress: F) -> Result<IterationReport>
    where
        F: FnMut(&TrainProgress),
    {
        self.step_with_callbacks(|_| {}, progress)
    }

    pub fn step_with_callbacks<SelfPlayProgressCallback, TrainProgressCallback>(
        &mut self,
        mut self_play_progress: SelfPlayProgressCallback,
        mut train_progress: TrainProgressCallback,
    ) -> Result<IterationReport>
    where
        SelfPlayProgressCallback: FnMut(SelfPlayProgress),
        TrainProgressCallback: FnMut(&TrainProgress),
    {
        self.step_with_callback_refs(&mut self_play_progress, &mut train_progress)
    }

    fn step_with_callback_refs<SelfPlayProgressCallback, TrainProgressCallback>(
        &mut self,
        self_play_progress: &mut SelfPlayProgressCallback,
        train_progress: &mut TrainProgressCallback,
    ) -> Result<IterationReport>
    where
        SelfPlayProgressCallback: FnMut(SelfPlayProgress),
        TrainProgressCallback: FnMut(&TrainProgress),
    {
        match &mut self.inner {
            TrainingRunKind::Connect4(run) => {
                run.step_with_callbacks(self_play_progress, train_progress)
            }
            TrainingRunKind::ChessClassic(run) => {
                run.step_with_callbacks(self_play_progress, train_progress)
            }
            TrainingRunKind::ChessH1(run) => {
                run.step_with_callbacks(self_play_progress, train_progress)
            }
            TrainingRunKind::ChessH4(run) => {
                run.step_with_callbacks(self_play_progress, train_progress)
            }
            TrainingRunKind::ChessH8(run) => {
                run.step_with_callbacks(self_play_progress, train_progress)
            }
        }
    }

    pub fn run(&mut self, limit: RunLimit) -> Result<()> {
        self.run_with_progress(limit, |_| {})
    }

    pub fn run_with_progress<F>(&mut self, limit: RunLimit, progress: F) -> Result<()>
    where
        F: FnMut(&TrainProgress),
    {
        self.run_with_callbacks(limit, |_| {}, progress)
    }

    pub fn run_with_callbacks<SelfPlayProgressCallback, TrainProgressCallback>(
        &mut self,
        limit: RunLimit,
        mut self_play_progress: SelfPlayProgressCallback,
        mut train_progress: TrainProgressCallback,
    ) -> Result<()>
    where
        SelfPlayProgressCallback: FnMut(SelfPlayProgress),
        TrainProgressCallback: FnMut(&TrainProgress),
    {
        match limit {
            RunLimit::Iterations(iterations) => {
                for _ in 0..iterations {
                    self.step_with_callback_refs(&mut self_play_progress, &mut train_progress)?;
                }
            }
            RunLimit::Forever => loop {
                self.step_with_callback_refs(&mut self_play_progress, &mut train_progress)?;
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

    /// Returns the outstanding TensorRT compilation requirement, if this
    /// process has trained a newer checkpoint than its fixed inference module.
    pub fn pending_tensor_rt_recompile(&self) -> Option<&NextInference> {
        match &self.inner {
            TrainingRunKind::Connect4(run) => run.pending_tensor_rt_recompile.as_ref(),
            TrainingRunKind::ChessClassic(run) => run.pending_tensor_rt_recompile.as_ref(),
            TrainingRunKind::ChessH1(run) => run.pending_tensor_rt_recompile.as_ref(),
            TrainingRunKind::ChessH4(run) => run.pending_tensor_rt_recompile.as_ref(),
            TrainingRunKind::ChessH8(run) => run.pending_tensor_rt_recompile.as_ref(),
        }
    }

    /// Loads a newly compiled TensorRT module and reconnects self-play to it.
    ///
    /// The configured module path must already contain the TensorRT export of
    /// the checkpoint reported by [`Self::pending_tensor_rt_recompile`]. This
    /// preserves replay and trainer state while replacing the fixed inference
    /// service and its evaluation cache.
    pub fn reload_tensor_rt_inference(&mut self) -> Result<()> {
        match &mut self.inner {
            TrainingRunKind::Connect4(run) => run.reload_tensor_rt_inference(),
            TrainingRunKind::ChessClassic(run) => run.reload_tensor_rt_inference(),
            TrainingRunKind::ChessH1(run) => run.reload_tensor_rt_inference(),
            TrainingRunKind::ChessH4(run) => run.reload_tensor_rt_inference(),
            TrainingRunKind::ChessH8(run) => run.reload_tensor_rt_inference(),
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
    fn step_with_callbacks<SelfPlayProgressCallback, TrainProgressCallback>(
        &mut self,
        self_play_progress: &mut SelfPlayProgressCallback,
        train_progress: &mut TrainProgressCallback,
    ) -> Result<IterationReport>
    where
        SelfPlayProgressCallback: FnMut(SelfPlayProgress),
        TrainProgressCallback: FnMut(&TrainProgress),
    {
        self.ensure_inference_is_current()?;

        let self_play = self.generate_self_play(self_play_progress)?;
        let training = self.train_network(&self_play, train_progress);

        let next_state = self.advance_state(&self_play, training.as_ref());
        let next_state = self.save_checkpoint(next_state)?;
        let refresh_plan = self.plan_inference_refresh()?;
        let next_inference = self.execute_inference_refresh(refresh_plan)?;

        self.state = next_state;
        let report = self.build_report(self_play, training, next_inference);
        self.persist_report(&report)?;

        Ok(report)
    }

    fn generate_self_play<Progress>(&self, progress: &mut Progress) -> Result<SelfPlayStats>
    where
        Progress: FnMut(SelfPlayProgress),
    {
        let epoch = SelfPlayEpoch {
            model_generation: self.state.model_generation,
            first_game_id: self.state.total_games_generated,
        };

        self.self_play
            .run_with_progress(&self.workers, &self.replay, epoch, progress)
    }

    fn train_network<Progress>(
        &mut self,
        self_play: &SelfPlayStats,
        progress: &mut Progress,
    ) -> Option<TrainMetrics>
    where
        Progress: FnMut(&TrainProgress),
    {
        self.trainer.train_with_progress(
            &self.network,
            &self.replay,
            &mut self.sampler,
            TrainingInvocation {
                device: self.device,
                seed: TrainingSeed {
                    experiment_seed: self.experiment.seed,
                    global_step: self.state.global_step,
                },
                fresh_replay_samples: self_play.moves,
            },
            progress,
        )
    }

    fn advance_state(
        &self,
        self_play: &SelfPlayStats,
        training: Option<&TrainMetrics>,
    ) -> RunState {
        advance_run_state(
            &self.state,
            IterationFacts {
                games: self_play.games,
                trained_steps: training.map_or(0, |metrics| metrics.train_steps),
                replay_samples: self.replay.len(),
            },
        )
    }

    fn save_checkpoint(&self, mut state: RunState) -> Result<RunState> {
        self.run_dir
            .write_latest(&mut state, |path| Ok(self.var_store.save(path)?))?;

        Ok(state)
    }

    fn ensure_inference_is_current(&self) -> Result<()> {
        if let Some(requirement) = &self.pending_tensor_rt_recompile {
            let NextInference::TensorRtRecompileRequired {
                checkpoint,
                compiled_artifact,
            } = requirement
            else {
                unreachable!("only TensorRT requirements are stored as pending")
            };

            anyhow::bail!(
                "TensorRT inference is stale after checkpoint {}; compile it into {} and call reload_tensor_rt_inference before another self-play generation",
                checkpoint.display(),
                compiled_artifact.display(),
            );
        }

        Ok(())
    }

    fn plan_inference_refresh(&self) -> Result<InferenceRefreshPlan> {
        let compiled_artifact = (self.experiment.inference.engine != InferenceEngine::Native)
            .then(|| compiled_artifact_path(&self.run_dir, &self.experiment))
            .transpose()?;

        Ok(plan_inference_refresh(
            self.experiment.inference.engine,
            self.run_dir.latest_path(),
            compiled_artifact,
        ))
    }

    fn execute_inference_refresh(&mut self, plan: InferenceRefreshPlan) -> Result<NextInference> {
        match plan {
            InferenceRefreshPlan::ReloadNative { checkpoint } => {
                self.inference
                    .reload_weights(&checkpoint)
                    .map_err(anyhow::Error::msg)?;

                Ok(NextInference::NativeReloaded)
            }
            InferenceRefreshPlan::RequireTensorRtRecompile {
                checkpoint,
                compiled_artifact,
            } => {
                let requirement = NextInference::TensorRtRecompileRequired {
                    checkpoint,
                    compiled_artifact,
                };
                self.pending_tensor_rt_recompile = Some(requirement.clone());

                Ok(requirement)
            }
        }
    }

    fn reload_tensor_rt_inference(&mut self) -> Result<()> {
        anyhow::ensure!(
            matches!(
                self.experiment.inference.engine,
                InferenceEngine::TensorRtTorchScript | InferenceEngine::TensorRtRaw
            ),
            "reload_tensor_rt_inference is only valid for TensorRT inference"
        );
        anyhow::ensure!(
            self.pending_tensor_rt_recompile.is_some(),
            "TensorRT inference is already current; run an iteration before reloading it"
        );

        let inference =
            load_inference_service(&self.run_dir, &self.experiment, &self.state, self.device)?;
        let workers = (self.make_workers)(&inference, self.experiment.self_play.clone())?;

        self.workers = workers;
        self.inference = inference;
        self.pending_tensor_rt_recompile = None;

        Ok(())
    }

    fn build_report(
        &self,
        self_play: SelfPlayStats,
        training: Option<TrainMetrics>,
        next_inference: NextInference,
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
            next_inference,
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
            "next_inference": &report.next_inference,
        }))
    }
}

fn advance_run_state(state: &RunState, facts: IterationFacts) -> RunState {
    let mut next = state.clone();
    next.total_games_generated += facts.games as u64;
    next.iteration += 1;
    next.model_generation += 1;
    next.global_step += facts.trained_steps as u64;
    next.replay_sample_count = facts.replay_samples;
    next.optimizer_moments_restored = false;

    next
}

fn plan_inference_refresh(
    engine: InferenceEngine,
    checkpoint: PathBuf,
    compiled_artifact: Option<PathBuf>,
) -> InferenceRefreshPlan {
    match engine {
        InferenceEngine::Native => InferenceRefreshPlan::ReloadNative { checkpoint },
        InferenceEngine::TensorRtTorchScript | InferenceEngine::TensorRtRaw => {
            InferenceRefreshPlan::RequireTensorRtRecompile {
                checkpoint,
                compiled_artifact: compiled_artifact
                    .expect("TensorRT refresh planning requires a compiled artifact path"),
            }
        }
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

fn load_inference_service(
    run_dir: &RunDir,
    experiment: &ExperimentConfig,
    state: &RunState,
    device: Device,
) -> Result<InferenceService> {
    match experiment.inference.engine {
        InferenceEngine::Native => {
            let checkpoint = run_dir
                .latest_checkpoint(state)
                .context("run has no current checkpoint")?;
            InferenceService::load(
                &experiment.model,
                InferenceSource::Checkpoint(&checkpoint),
                device,
                &experiment.inference,
            )
        }
        InferenceEngine::TensorRtTorchScript => {
            let artifact = compiled_artifact_path(run_dir, experiment)?;
            validate_checkpoint_identity(&artifact, state)?;

            InferenceService::load(
                &experiment.model,
                InferenceSource::TensorRtTorchScript(&artifact),
                device,
                &experiment.inference,
            )
        }
        InferenceEngine::TensorRtRaw => {
            let artifact = compiled_artifact_path(run_dir, experiment)?;
            validate_checkpoint_identity(&artifact, state)?;

            InferenceService::load(
                &experiment.model,
                InferenceSource::TensorRtEngine(&artifact),
                device,
                &experiment.inference,
            )
        }
    }
}

fn compiled_artifact_path(run_dir: &RunDir, experiment: &ExperimentConfig) -> Result<PathBuf> {
    let artifact = experiment
        .inference
        .compiled_artifact
        .as_ref()
        .expect("validated TensorRT inference configuration");
    let artifact = if artifact.is_absolute() {
        artifact.clone()
    }
    else {
        run_dir.root().join(artifact)
    };

    anyhow::ensure!(
        artifact.is_file(),
        "missing TensorRT compiled artifact: {}",
        artifact.display()
    );

    Ok(artifact)
}

fn validate_checkpoint_identity(artifact: &Path, state: &RunState) -> Result<()> {
    let checkpoint = state
        .checkpoint_identity
        .as_ref()
        .context("run has no current checkpoint identity")?;
    let manifest = crate::artifact::read_manifest(artifact)?;
    if manifest.checkpoint_sha256 != checkpoint.sha256 {
        return Err(crate::artifact::RecompileRequired {
            reason: crate::artifact::RecompileReason::StaleIdentity("checkpoint_sha256"),
        }
        .into());
    }

    Ok(())
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
            make_workers: DomainSelfPlayWorkerFactory::new,
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
            make_workers: DomainSelfPlayWorkerFactory::new,
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
            make_workers: DomainSelfPlayWorkerFactory::new,
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
            make_workers,
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
            make_workers,
            trainer,
            sampler: ReplaySampler::new(representation),
            pending_tensor_rt_recompile: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{CheckpointIdentity, ResumeKind, STATE_FORMAT_VERSION};

    fn populated_state() -> RunState {
        RunState {
            format_version: STATE_FORMAT_VERSION,
            iteration: 4,
            model_generation: 7,
            global_step: 11,
            total_games_generated: 13,
            checkpoint_identity: Some(CheckpointIdentity {
                generation: 7,
                relative_path: "checkpoints/latest.safetensors".into(),
                sha256: "abc".into(),
            }),
            replay_sample_count: 17,
            optimizer_moments_restored: true,
            resume_kind: ResumeKind::Full,
        }
    }

    #[test]
    fn advancing_run_state_updates_iteration_facts_and_preserves_durable_fields() {
        let state = populated_state();
        let next = advance_run_state(
            &state,
            IterationFacts {
                games: 3,
                trained_steps: 5,
                replay_samples: 19,
            },
        );

        assert_eq!(next.iteration, 5);
        assert_eq!(next.model_generation, 8);
        assert_eq!(next.global_step, 16);
        assert_eq!(next.total_games_generated, 16);
        assert_eq!(next.replay_sample_count, 19);
        assert!(!next.optimizer_moments_restored);
        assert_eq!(next.format_version, state.format_version);
        assert_eq!(next.checkpoint_identity, state.checkpoint_identity);
        assert_eq!(next.resume_kind, state.resume_kind);
    }

    #[test]
    fn advancing_run_state_without_training_keeps_global_step() {
        let state = populated_state();
        let next = advance_run_state(
            &state,
            IterationFacts {
                games: 2,
                trained_steps: 0,
                replay_samples: 23,
            },
        );

        assert_eq!(next.global_step, state.global_step);
    }

    #[test]
    fn native_refresh_plan_reloads_the_checkpoint() {
        assert_eq!(
            plan_inference_refresh(
                InferenceEngine::Native,
                PathBuf::from("checkpoints/latest.safetensors"),
                None,
            ),
            InferenceRefreshPlan::ReloadNative {
                checkpoint: PathBuf::from("checkpoints/latest.safetensors"),
            }
        );
    }

    #[test]
    fn tensor_rt_refresh_plan_keeps_exact_paths() {
        assert_eq!(
            plan_inference_refresh(
                InferenceEngine::TensorRtRaw,
                PathBuf::from("checkpoints/latest.safetensors"),
                Some(PathBuf::from("artifacts/model.plan")),
            ),
            InferenceRefreshPlan::RequireTensorRtRecompile {
                checkpoint: PathBuf::from("checkpoints/latest.safetensors"),
                compiled_artifact: PathBuf::from("artifacts/model.plan"),
            }
        );
    }

    #[test]
    fn tensor_rt_requirement_identifies_the_checkpoint_and_module() {
        let requirement = NextInference::TensorRtRecompileRequired {
            checkpoint: PathBuf::from("checkpoints/latest.safetensors"),
            compiled_artifact: PathBuf::from("model.trt.ts"),
        };

        let value = serde_json::to_value(&requirement).unwrap();

        assert_eq!(value["kind"], "tensor-rt-recompile-required");
        assert_eq!(value["checkpoint"], "checkpoints/latest.safetensors");
        assert_eq!(value["compiled_artifact"], "model.trt.ts");
    }
}
