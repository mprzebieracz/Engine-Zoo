//! Neural AlphaZero training and inference.

mod analysis;
pub mod artifact;
mod batcher;
mod evaluator;
pub mod experiment;
pub mod inference;
mod network;
mod replay;
pub mod representation;
mod rules;
pub mod runtime;
mod selfplay;
mod trainer;
mod training_run;

pub use analysis::{
    analyze_game_mcts, analyze_game_mcts_with_repetitions, analyze_game_net, Analysis,
    AnalyzeConfig, AnalyzeMode,
};
pub use artifact::{
    ArtifactBuildConfig, ArtifactCacheIdentity, ArtifactDType, ArtifactEnvironment,
    ArtifactIoContract, CompiledArtifactManifest, CompiledBackendKind, CompilerIdentity,
    RecompileReason, RecompileRequired, TensorRtBuildPrecision, ValueLayout,
    ARTIFACT_MANIFEST_VERSION,
};
#[cfg(feature = "raw-tensorrt")]
pub use batcher::RawTensorRtBackend;
pub use batcher::{
    Batcher, BatcherClient, BatcherConfig, BatcherError, BatcherStats, CombinedEncodedBatch,
    InferenceBackend, InferencePrecision, TchInferenceBackend,
};
pub use evaluator::{EncodedEvalBatch, EncodedEvaluator, RepresentedEvaluator};
pub use experiment::{
    CheckpointIdentity, DurationConfig, ExperimentConfig, InferenceConfig, InferenceEngine,
    ReplayConfig, ResumeKind, RunDir, RunState, EXPERIMENT_FORMAT_VERSION, STATE_FORMAT_VERSION,
};
pub use inference::{InferenceClient, InferenceService, InferenceSource};
pub use network::{
    ChessHistory, GameKind, ModelFingerprint, ModelSpec, Network, RawNetworkOutput, RawValueOutput,
    ValueHeadSpec,
};
pub use replay::{
    Outcome, ReplayBatch, ReplayBuffer, ReplaySample, SampleMetadata, SearchKind, SparsePolicy,
    SparsePolicyBatch, TrainingWeights,
};
pub use representation::{Action, AlphaZeroRepresentation};
pub use rules::{ChessNodeMeta, ChessRepetitionRules};
pub use runtime::ChessAlphaZeroEngine;
pub(crate) use search::{EvalTable, Mcts, SearchAlgorithm, SearchConfig, SearchRequest};
pub use selfplay::{
    select_temperature_action, ChessCanonicalSelfPlayDomain, ChessClassicSelfPlayDomain,
    CompletedGame, DomainSelfPlayWorkerFactory, GameRequest, GumbelMoveSelection,
    ResignationConfig, SearchBudget, SearchBudgetSchedule, SelfPlayConfig, SelfPlayCoordinator,
    SelfPlayDomain, SelfPlayEpoch, SelfPlayStats, SelfPlayWorker, SelfPlayWorkerFactory,
    StandardSelfPlayDomain, TemperaturePhase, TemperatureSchedule,
};
pub use trainer::{
    build_optimizer, train, wdl_cross_entropy, LearningRateSchedule, OptimizerSpec, TrainConfig,
    TrainMetrics, TrainProgress, Trainer, TrainingSeed,
};
pub use training_run::{IterationReport, NextInference, RunLimit, TrainingRun};
