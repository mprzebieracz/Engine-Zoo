//! The AlphaZero algorithm. Trained from its own self-play games.

mod batcher;
mod checkpoint;
mod evaluator;
mod mcts;
mod network;
mod replay;
mod selfplay;
mod trainer;

pub use batcher::{Batcher, BatcherClient, BatcherStats, InferencePrecision};
pub use checkpoint::{RunArchitecture, RunConfig, RunDir};
pub use engine_core::rules::RepetitionGame;
pub use evaluator::{EvalBatch, Evaluation, Evaluator};
pub use mcts::{
    EvalTable, EvalTableStats, GumbelSearchProfile, Mcts, MctsConfig, MctsVariant, SearchResult,
};
pub use network::{
    wdl_scalar, AlphaZeroNet, ChessAzV2Config, ChessAzV2Net, LegacyAlphaZeroNet, NetConfig,
    Network, NetworkConfig, NetworkOutput,
};
pub use replay::{ReplayBatch, ReplayBuffer, SparsePolicyBatch, Transition};
pub use selfplay::{
    select_temperature_action, self_play, ChessV2GumbelProfiles, SelfPlayConfig, SelfPlayStats,
    SelfPlayTemperature,
};
pub use trainer::{
    build_optimizer, train, train_chess_az_v2, wdl_cross_entropy, TrainConfig, TrainMetrics,
};
