//! Neural AlphaZero training and inference.

mod analysis;
mod batcher;
mod checkpoint;
mod evaluator;
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
pub use batcher::{Batcher, BatcherClient, BatcherStats, InferencePrecision};
pub use checkpoint::{
    ChessScalarAzV1Config, Connect4ScalarAzConfig, ModelConfig, RunConfig, RunDir,
    RUN_CONFIG_FORMAT_VERSION,
};
pub use evaluator::{EncodedEvalBatch, EncodedEvaluator, RepresentedEvaluator};
pub use network::{
    wdl_scalar, AlphaZeroNet, ChessAzV2Config, ChessAzV2Net, LegacyAlphaZeroNet, NetConfig,
    Network, NetworkConfig, NetworkOutput,
};
pub use replay::{ReplayBatch, ReplayBuffer, SparsePolicyBatch, Transition};
pub use representation::{Action, AlphaZeroRepresentation};
pub use rules::{ChessNodeMeta, ChessRepetitionRules};
pub use search::{
    EvalTable, EvalTableStats, Evaluation, GumbelSearchProfile, Mcts, MctsConfig, MctsVariant,
    NoExtraRules, PolicyValueEvaluator, RuleResult, SearchResult, SearchRules,
};
pub use selfplay::{
    select_temperature_action, self_play, ChessV2GumbelProfiles, SelfPlayConfig, SelfPlayStats,
    SelfPlayTemperature,
};
pub use trainer::{
    build_optimizer, train, train_chess_az_v2, wdl_cross_entropy, TrainConfig, TrainMetrics,
};
