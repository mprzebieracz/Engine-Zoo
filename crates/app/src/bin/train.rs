use alphazero::representation::Connect4AzRepresentation;
use alphazero::{
    self_play, train, train_chess_az_v2, AlphaZeroNet, Batcher, ChessAzV2Config, ChessAzV2Net,
    ChessScalarAzV1Config, ChessV2GumbelProfiles, Connect4ScalarAzConfig, GumbelSearchProfile,
    InferencePrecision, MctsConfig, MctsVariant, ModelConfig, NetworkConfig, ReplayBuffer,
    RunConfig, RunDir, SelfPlayConfig, SelfPlayStats, TrainConfig, RUN_CONFIG_FORMAT_VERSION,
};
use anyhow::Result;
use checkpoint_eval::arena::{self, ArenaConfig};
use clap::{Parser, ValueEnum};
use engine_app::chess_selfplay::{self_play_chess, self_play_chess_az_v2};
use games::{ChessGame, Connect4};
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tch::{nn, Device};

#[path = "train/legacy.rs"]
mod legacy;
#[path = "train/v2.rs"]
mod v2;

use legacy::{run, save_numbered_checkpoint};
use v2::run_chess_az_v2;

#[derive(Clone, Copy, ValueEnum)]
enum GameKind {
    Connect4,
    Chess,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ArchitectureKind {
    Legacy,
    ChessAzV2,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    /// Every iteration's network becomes the self-play network.
    Continuous,
    /// A candidate replaces the self-play network only after beating it in an
    /// arena match.
    Gated,
}

#[derive(Clone, Copy, ValueEnum)]
enum SearchKind {
    /// Standard AlphaZero PUCT search.
    Puct,
    /// Gumbel AlphaZero-style root sampling and sequential halving.
    Gumbel,
}

impl From<SearchKind> for MctsVariant {
    fn from(value: SearchKind) -> Self {
        match value {
            SearchKind::Puct => MctsVariant::Puct,
            SearchKind::Gumbel => MctsVariant::Gumbel {
                sampled_actions: 16,
            },
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceKind {
    /// Use CUDA when available, otherwise CPU.
    Auto,
    /// Require a CUDA device.
    Cuda,
    /// Force CPU.
    Cpu,
}

#[derive(Clone, Copy, ValueEnum)]
enum InferencePrecisionKind {
    /// FP16 on CUDA, FP32 on CPU.
    Auto,
    Fp32,
    Fp16,
}

#[derive(Parser)]
#[command(about = "AlphaZero self-play + training loop")]
struct Args {
    #[arg(long, value_enum)]
    game: GameKind,
    /// Network/data format for a new run. Existing runs always use their saved architecture.
    #[arg(long, value_enum)]
    architecture: Option<ArchitectureKind>,
    /// Chess AZ v2 history frames for a new run (1, 4, or 8; default 4).
    #[arg(long)]
    history: Option<usize>,
    /// Run directory (checkpoints, metrics); defaults to data/runs/<game>.
    #[arg(long)]
    run_dir: Option<PathBuf>,
    /// File for C++/libtorch stderr output. Defaults to <run-dir>/stderr.log.
    #[arg(long)]
    stderr_log: Option<PathBuf>,
    /// Print self-play progress every N completed games. Set 0 to disable.
    #[arg(long, default_value_t = 25)]
    progress_every: usize,

    #[arg(long, default_value_t = 10)]
    iterations: usize,
    /// Ignore --iterations and keep training until interrupted.
    #[arg(long)]
    forever: bool,
    /// Save an extra timestamped checkpoint this often. Set 0 to disable.
    #[arg(long, default_value_t = 60)]
    archive_checkpoint_minutes: u64,
    /// Save checkpoints/ckpt_NNNN every N promoted/continuous updates. best.safetensors is still saved every update.
    /// Set 1 to save every update. Set 0 to disable numbered checkpoints after ckpt_0000.
    #[arg(long, default_value_t = 1)]
    numbered_checkpoint_every: u32,
    #[arg(long, default_value_t = 100)]
    games: usize,
    #[arg(long, default_value = "32")]
    threads: Option<usize>,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    /// Leaf evaluations collected inside one tree search before evaluator calls.
    #[arg(long, default_value_t = 32)]
    mcts_leaf_batch_size: usize,
    /// Batcher states to coalesce before running network inference.
    #[arg(long)]
    wait_for: Option<usize>,
    /// Batcher timeout before running a partial inference batch.
    #[arg(long, default_value_t = 5)]
    batch_timeout_ms: u64,
    /// Self-play inference precision. Training always stays FP32.
    #[arg(long, value_enum, default_value_t = InferencePrecisionKind::Auto)]
    inference_precision: InferencePrecisionKind,
    /// Chess self-play transposition table entries. Set 0 to disable.
    #[arg(long, default_value_t = 1_000_000)]
    tt_entries: usize,
    /// Fast-search simulations for playout cap randomization.
    #[arg(long)]
    fast_simulations: Option<usize>,
    /// Probability of using the full simulation count for a move.
    #[arg(long, default_value_t = 0.25)]
    full_simulation_probability: f32,
    /// Chess AZ v2 full-search simulation budget.
    #[arg(long, default_value_t = 128)]
    v2_full_simulations: usize,
    /// Chess AZ v2 sequential-halving candidates for the full budget.
    #[arg(long, default_value_t = 16)]
    v2_full_root_candidates: usize,
    /// Chess AZ v2 fast-search simulation budget.
    #[arg(long, default_value_t = 64)]
    v2_fast_simulations: usize,
    /// Chess AZ v2 sequential-halving candidates for the fast budget.
    #[arg(long, default_value_t = 8)]
    v2_fast_root_candidates: usize,
    /// Probability of selecting the full Chess AZ v2 Gumbel budget.
    #[arg(long, default_value_t = 0.5)]
    v2_full_simulation_probability: f32,
    #[arg(long)]
    disable_resignation: bool,
    #[arg(long, default_value_t = -0.95)]
    resignation_threshold: f32,
    #[arg(long, default_value_t = 60)]
    resignation_min_ply: usize,
    #[arg(long, default_value_t = 3)]
    resignation_consecutive_moves: usize,
    #[arg(long, default_value_t = 0.10)]
    resignation_disable_probability: f32,
    #[arg(long, value_enum, default_value_t = SearchKind::Gumbel)]
    mcts_variant: SearchKind,
    /// Root actions considered by Gumbel MCTS before sequential halving.
    #[arg(long, default_value_t = 16)]
    gumbel_sampled_actions: usize,
    /// First Play Urgency reduction for unvisited MCTS children.
    #[arg(long, default_value_t = 0.1)]
    fpu_reduction: f32,
    #[arg(long, default_value_t = 512)]
    max_moves: usize,

    #[arg(long, default_value_t = 500_000)]
    buffer: usize,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 4096)]
    minibatch_size: usize,
    #[arg(long, default_value_t = 80)]
    train_steps: usize,
    /// Print training progress every N optimizer steps. Set 0 to disable.
    #[arg(long, default_value_t = 10)]
    train_progress_every: usize,
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    #[arg(long, default_value_t = 1e-4)]
    weight_decay: f64,

    /// Residual blocks for a freshly created network (existing runs keep theirs).
    #[arg(long)]
    blocks: Option<i64>,
    /// Conv filters for a freshly created network.
    #[arg(long)]
    filters: Option<i64>,

    #[arg(long, value_enum, default_value_t = Mode::Continuous)]
    mode: Mode,
    #[arg(long, default_value_t = 40)]
    gate_games: usize,
    #[arg(long, default_value_t = 0.55)]
    gate_threshold: f32,
    #[arg(long, default_value_t = 400)]
    gate_simulations: usize,

    #[arg(long, value_enum, default_value_t = DeviceKind::Cuda)]
    device: DeviceKind,
}

fn default_cpu_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// How long the batcher waits for more states before running a partial batch.
const BATCH_TIMEOUT: Duration = Duration::from_millis(2);

fn main() -> Result<()> {
    let args = Args::parse();
    redirect_stderr(&args)?;
    match args.game {
        GameKind::Connect4 => {
            run::<Connect4>(args, 5, 64, self_play::<Connect4, Connect4AzRepresentation>)
        }
        GameKind::Chess => run_chess(args),
    }
}

fn default_run_root(args: &Args) -> PathBuf {
    args.run_dir.clone().unwrap_or_else(|| {
        let game = match args.game {
            GameKind::Connect4 => "connect4".to_owned(),
            GameKind::Chess => match args.architecture.unwrap_or(ArchitectureKind::ChessAzV2) {
                ArchitectureKind::Legacy => "chess".to_owned(),
                ArchitectureKind::ChessAzV2 => {
                    format!("chess-az-v2-h{}", args.history.unwrap_or(4))
                }
            },
        };
        PathBuf::from("runs").join(game)
    })
}

fn redirect_stderr(args: &Args) -> Result<()> {
    let path = args
        .stderr_log
        .clone()
        .unwrap_or_else(|| default_run_root(args).join("stderr.log"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
    unsafe {
        anyhow::ensure!(
            libc::dup2(file.as_raw_fd(), libc::STDERR_FILENO) >= 0,
            "failed to redirect stderr to {}",
            path.display()
        );
    }
    println!("stderr log: {}", path.display());
    Ok(())
}

fn run_chess(mut args: Args) -> Result<()> {
    let root = default_run_root(&args);
    let requested_architecture = args.architecture;
    let requested_history = args.history;
    let (run_dir, cfg) = RunDir::open_or_create(&root, || {
        let model = match requested_architecture.unwrap_or(ArchitectureKind::ChessAzV2) {
            ArchitectureKind::Legacy => ModelConfig::ChessScalarAzV1(ChessScalarAzV1Config {
                num_res_blocks: args.blocks.unwrap_or(10),
                num_filters: args.filters.unwrap_or(64),
            }),
            ArchitectureKind::ChessAzV2 => ModelConfig::ChessAzV2(ChessAzV2Config {
                history: requested_history.unwrap_or(4),
            }),
        };
        RunConfig {
            format_version: RUN_CONFIG_FORMAT_VERSION,
            model,
        }
    })?;
    anyhow::ensure!(
        cfg.model.game_name() == "chess",
        "run {} is not a chess run",
        root.display()
    );
    if let Some(requested) = requested_architecture {
        let actual = match cfg.model {
            ModelConfig::ChessScalarAzV1(_) => ArchitectureKind::Legacy,
            ModelConfig::ChessAzV2(_) => ArchitectureKind::ChessAzV2,
            ModelConfig::Connect4ScalarAz(_) => unreachable!("validated chess run"),
        };
        anyhow::ensure!(
            requested == actual,
            "--architecture contradicts saved run config"
        );
    }
    match cfg.model.clone() {
        ModelConfig::ChessScalarAzV1(_) => {
            args.run_dir = Some(root);
            run::<ChessGame>(args, 10, 64, self_play_chess)
        }
        ModelConfig::ChessAzV2(v2) => {
            if let Some(history) = requested_history {
                anyhow::ensure!(
                    history == v2.history,
                    "--history contradicts saved run config"
                );
            }
            run_chess_az_v2(args, run_dir, cfg, v2)
        }
        ModelConfig::Connect4ScalarAz(_) => unreachable!("validated chess run"),
    }
}
