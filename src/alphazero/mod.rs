//! The AlphaZero algorithm. Trained from its own self-play games.

mod batcher;
mod checkpoint;
mod evaluator;
mod mcts;
mod network;
mod replay;
mod selfplay;
mod trainer;

pub use batcher::{Batcher, BatcherClient, InferencePrecision};
pub use checkpoint::{RunConfig, RunDir};
pub use evaluator::{EvalBatch, Evaluation, Evaluator};
pub use mcts::{Mcts, MctsConfig, MctsVariant, RepetitionGame, SearchResult};
pub use network::{AlphaZeroNet, NetConfig};
pub use replay::{ReplayBuffer, Transition};
pub use selfplay::{self_play, self_play_chess, SelfPlayConfig, SelfPlayStats};
pub use trainer::{build_optimizer, train, TrainConfig, TrainMetrics};
