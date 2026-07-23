//! Neural AlphaZero training and inference.

mod analysis;
mod batcher;
mod evaluator;
pub mod experiment;
mod network;
mod replay;
pub mod representation;
mod rules;
mod selfplay;
mod trainer;

pub use analysis::{
    analyze_game_mcts, analyze_game_mcts_with_repetitions, analyze_game_net, Analysis,
    AnalyzeConfig, AnalyzeMode,
};
pub use batcher::{
    Batcher, BatcherClient, BatcherConfig, BatcherError, BatcherStats, CombinedEncodedBatch,
    InferenceBackend, InferencePrecision, TchInferenceBackend,
};
pub use evaluator::{EncodedEvalBatch, EncodedEvaluator, RepresentedEvaluator};
pub use experiment::{
    DurationConfig, ExperimentConfig, InferenceConfig, InferenceEngine, ReplayConfig, ResumeKind,
    RunDir, RunState, EXPERIMENT_FORMAT_VERSION, STATE_FORMAT_VERSION,
};
pub use network::{
    ChessHistoryLength, ClassicResidualNet, GameSpec, ModelFingerprint, ModelSpec, Network,
    NetworkSpec, PolicyHeadConfig, RawNetworkOutput, RawValueOutput, RepresentationSpec,
    ResidualNetworkConfig, ResidualTrunkConfig, SeResidualNet, SeTrunkSpec, ValueHeadConfig,
};
pub use replay::{
    Outcome, ReplayBatch, ReplayBuffer, ReplaySample, SampleMetadata, SearchKind, SparsePolicy,
    SparsePolicyBatch, TrainingWeights,
};
pub use representation::{Action, AlphaZeroRepresentation};
pub use rules::{ChessNodeMeta, ChessRepetitionRules};
pub use search::{
    CommonSearchConfig, CompletedQConfig, DirichletConfig, EvalTable, EvalTableStats, Evaluation,
    EvaluationError, FpuConfig, GumbelConfig, InFlightConfig, Mcts, NoExtraRules,
    PolicyValueEvaluator, PuctConfig, PuctSelectionConfig, RuleResult, SearchConfig,
    SearchConfigError, SearchDiagnostics, SearchError, SearchResult, SearchRules,
};
pub use selfplay::{
    select_temperature_action, ChessSelfPlayWorkerFactory, CompletedGame, GameRequest,
    GenericSelfPlayWorkerFactory, GumbelMoveSelection, ResignationConfig, SearchBudget,
    SearchBudgetSchedule, SelfPlayConfig, SelfPlayCoordinator, SelfPlayEpoch, SelfPlayStats,
    SelfPlayWorker, SelfPlayWorkerFactory, TemperaturePhase, TemperatureSchedule,
};
pub use trainer::{
    build_optimizer, train, wdl_cross_entropy, LearningRateSchedule, OptimizerSpec, TrainConfig,
    TrainMetrics,
};
