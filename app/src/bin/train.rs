use algorithms::alphazero::{
    self_play, self_play_chess, train, AlphaZeroNet, Batcher, InferencePrecision, Mcts, MctsConfig,
    MctsVariant, NetConfig, ReplayBuffer, RunConfig, RunDir, SelfPlayConfig, SelfPlayStats,
    TrainConfig,
};
use anyhow::Result;
use clap::{Parser, ValueEnum};
use engine_core::arena::{self, ArenaConfig};
use engine_core::game::Game;
use games::{ChessGame, Connect4};
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tch::{nn, Device};

#[derive(Clone, Copy, ValueEnum)]
enum GameKind {
    Connect4,
    Chess,
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
    /// Run directory (checkpoints, metrics); defaults to runs/<game>.
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
        GameKind::Connect4 => run::<Connect4>(args, 5, 64, self_play::<Connect4>),
        GameKind::Chess => run::<ChessGame>(args, 10, 64, self_play_chess),
    }
}

fn default_run_root(args: &Args) -> PathBuf {
    args.run_dir.clone().unwrap_or_else(|| {
        let game = match args.game {
            GameKind::Connect4 => "connect4",
            GameKind::Chess => "chess",
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

fn run<G: Game>(
    args: Args,
    default_blocks: i64,
    default_filters: i64,
    self_play_fn: fn(&Batcher, &ReplayBuffer, &SelfPlayConfig) -> SelfPlayStats,
) -> Result<()> {
    let device = match args.device {
        DeviceKind::Auto => Device::cuda_if_available(),
        DeviceKind::Cuda => {
            anyhow::ensure!(
                tch::Cuda::is_available(),
                "CUDA requested but no GPU is available (check libtorch CUDA build and drivers)"
            );
            Device::Cuda(0)
        }
        DeviceKind::Cpu => Device::Cpu,
    };
    println!("device: {device:?}");
    if device.is_cuda() {
        tch::Cuda::cudnn_set_benchmark(true);
    }

    let threads = args.threads.unwrap_or_else(|| {
        if device.is_cuda() {
            128
        } else {
            default_cpu_threads()
        }
    });
    let wait_for = args
        .wait_for
        .unwrap_or_else(|| if device.is_cuda() { threads.min(24) } else { 1 });
    let self_play_precision = match args.inference_precision {
        InferencePrecisionKind::Auto => {
            if device.is_cuda() {
                InferencePrecision::Fp16
            } else {
                InferencePrecision::Fp32
            }
        }
        InferencePrecisionKind::Fp32 => InferencePrecision::Fp32,
        InferencePrecisionKind::Fp16 => InferencePrecision::Fp16,
    };

    let root = args
        .run_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("runs").join(G::NAME));
    let (run, cfg) = RunDir::open_or_create(&root, || RunConfig {
        game: G::NAME.into(),
        net: NetConfig::for_game::<G>(
            args.blocks.unwrap_or(default_blocks),
            args.filters.unwrap_or(default_filters),
        ),
    })?;
    anyhow::ensure!(
        cfg.game == G::NAME,
        "run dir {} holds a {} run, not {}",
        root.display(),
        cfg.game,
        G::NAME
    );

    let mut vs = nn::VarStore::new(device);
    let net = AlphaZeroNet::new(&vs.root(), &cfg.net);
    let mut next_ckpt = match run.latest_checkpoint() {
        Some((idx, path)) => {
            println!("resuming from {}", path.display());
            vs.load(&path)?;
            idx + 1
        }
        None => {
            vs.save(run.checkpoint_path(0))?;
            1
        }
    };
    if !run.best_path().exists() {
        vs.save(run.best_path())?;
    }

    let replay = ReplayBuffer::new(args.buffer, G::state_size(), G::ACTION_SIZE);
    let train_cfg = TrainConfig {
        micro_batch_size: args.batch_size,
        batch_size: args.minibatch_size,
        train_steps: args.train_steps,
        progress_every: args.train_progress_every,
        lr: args.lr,
        weight_decay: args.weight_decay,
    };
    let mut opt = algorithms::alphazero::build_optimizer(&vs, &train_cfg)?;
    let sp_cfg = SelfPlayConfig {
        num_games: args.games,
        threads,
        max_moves: args.max_moves,
        progress_every: args.progress_every,
        tt_entries: args.tt_entries,
        fast_simulations: args
            .fast_simulations
            .unwrap_or_else(|| (args.simulations / 8).max(1)),
        full_simulation_probability: args.full_simulation_probability,
        resignation_enabled: !args.disable_resignation,
        resignation_threshold: args.resignation_threshold,
        resignation_consecutive_moves: args.resignation_consecutive_moves,
        resignation_min_ply: args.resignation_min_ply,
        resignation_disable_probability: args.resignation_disable_probability,
        mcts: MctsConfig {
            simulations: args.simulations,
            leaf_batch_size: args.mcts_leaf_batch_size.max(1),
            fpu_reduction: args.fpu_reduction,
            variant: match args.mcts_variant {
                SearchKind::Puct => MctsVariant::Puct,
                SearchKind::Gumbel => MctsVariant::Gumbel {
                    sampled_actions: args.gumbel_sampled_actions,
                },
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let archive_interval = (args.archive_checkpoint_minutes > 0)
        .then(|| Duration::from_secs(args.archive_checkpoint_minutes * 60));
    let mut last_archive = Instant::now();
    let self_play_batcher = Batcher::new_with_precision(
        &cfg.net,
        &run.best_path(),
        device,
        wait_for,
        Duration::from_millis(args.batch_timeout_ms),
        self_play_precision,
    )?;
    let mut iteration = 0usize;
    while args.forever || iteration < args.iterations {
        println!("=== iteration {iteration} ===");
        let self_play_started = Instant::now();
        let self_play_stats = self_play_fn(&self_play_batcher, &replay, &sp_cfg);
        let self_play_secs = self_play_started.elapsed().as_secs_f64();
        let tt_queries = self_play_stats.tt_hits + self_play_stats.tt_misses;
        let tt_hit_rate = if tt_queries == 0 {
            0.0
        } else {
            self_play_stats.tt_hits as f64 / tt_queries as f64
        };
        println!(
            "self-play: {} games in {self_play_secs:.1}s ({:.1} games/s, {:.1} moves/game)",
            args.games,
            args.games as f64 / self_play_secs,
            self_play_stats.avg_moves_per_game()
        );
        println!(
            "self-play stats: full={} fast={} resignations={} tt={:.1}%",
            self_play_stats.full_searches,
            self_play_stats.fast_searches,
            self_play_stats.resignations,
            100.0 * tt_hit_rate
        );

        let train_started = Instant::now();
        let metrics = train(&net, &mut opt, &replay, device, &cfg.net, &train_cfg);
        let train_secs = train_started.elapsed().as_secs_f64();
        println!("train: {train_secs:.1}s");

        let mut record = json!({
            "iteration": iteration,
            "mode": match args.mode { Mode::Continuous => "continuous", Mode::Gated => "gated" },
            "mcts_variant": match args.mcts_variant { SearchKind::Puct => "puct", SearchKind::Gumbel => "gumbel" },
            "buffer_size": replay.len(),
            "self_play_secs": self_play_secs,
            "train_secs": train_secs,
            "games": args.games,
            "self_play_games": self_play_stats.games,
            "self_play_moves": self_play_stats.moves,
            "avg_moves_per_game": self_play_stats.avg_moves_per_game(),
            "full_searches": self_play_stats.full_searches,
            "fast_searches": self_play_stats.fast_searches,
            "resignations": self_play_stats.resignations,
            "tt_hits": self_play_stats.tt_hits,
            "tt_misses": self_play_stats.tt_misses,
            "tt_inserts": self_play_stats.tt_inserts,
            "tt_hit_rate": tt_hit_rate,
            "fpu_reduction": args.fpu_reduction,
            "mcts_leaf_batch_size": args.mcts_leaf_batch_size.max(1),
            "threads": threads,
            "wait_for": wait_for,
            "batch_timeout_ms": args.batch_timeout_ms,
            "tt_entries": args.tt_entries,
            "fast_simulations": sp_cfg.fast_simulations,
            "full_simulation_probability": sp_cfg.full_simulation_probability,
            "resignation_enabled": sp_cfg.resignation_enabled,
            "resignation_threshold": sp_cfg.resignation_threshold,
            "resignation_min_ply": sp_cfg.resignation_min_ply,
            "resignation_consecutive_moves": sp_cfg.resignation_consecutive_moves,
            "resignation_disable_probability": sp_cfg.resignation_disable_probability,
            "self_play_precision": match self_play_precision { InferencePrecision::Fp32 => "fp32", InferencePrecision::Fp16 => "fp16" },
        });
        if let Some(m) = &metrics {
            record["policy_loss"] = m.policy_loss.into();
            record["value_loss"] = m.value_loss.into();
            record["train_steps"] = m.train_steps.into();
        }

        match args.mode {
            Mode::Continuous => {
                let checkpoint_started = Instant::now();
                vs.save(run.checkpoint_path(next_ckpt))?;
                vs.save(run.best_path())?;
                self_play_batcher.reload_weights(&run.best_path())?;
                next_ckpt += 1;
                record["checkpoint_secs"] = checkpoint_started.elapsed().as_secs_f64().into();
            }
            Mode::Gated => {
                vs.save(run.candidate_path())?;
                let arena_cfg = ArenaConfig {
                    games: args.gate_games,
                    max_moves: args.max_moves,
                    ..Default::default()
                };
                let mcts_cfg = MctsConfig {
                    simulations: args.gate_simulations,
                    leaf_batch_size: args.mcts_leaf_batch_size.max(1),
                    eps: 0.0, // no exploration noise in evaluation play
                    variant: MctsVariant::Puct,
                    ..Default::default()
                };
                let wait_for = 1;
                let candidate = Batcher::new(
                    &cfg.net,
                    &run.candidate_path(),
                    device,
                    wait_for,
                    BATCH_TIMEOUT,
                )?;
                let baseline =
                    Batcher::new(&cfg.net, &run.best_path(), device, wait_for, BATCH_TIMEOUT)?;
                let winrate = arena::evaluate::<G>(
                    &mut Mcts::new(candidate.client(), mcts_cfg),
                    &mut Mcts::new(baseline.client(), mcts_cfg),
                    &arena_cfg,
                );

                let promoted = winrate >= args.gate_threshold;
                record["arena_winrate"] = winrate.into();
                record["gate_threshold"] = args.gate_threshold.into();
                record["promoted"] = promoted.into();
                if promoted {
                    println!(
                        "candidate promoted: {:.1}% >= {:.1}%",
                        100.0 * winrate,
                        100.0 * args.gate_threshold
                    );
                    vs.save(run.checkpoint_path(next_ckpt))?;
                    vs.save(run.best_path())?;
                    self_play_batcher.reload_weights(&run.best_path())?;
                    next_ckpt += 1;
                } else {
                    println!(
                        "candidate rejected: {:.1}% < {:.1}%; self-play keeps the old best",
                        100.0 * winrate,
                        100.0 * args.gate_threshold
                    );
                }
            }
        }

        if let Some(interval) = archive_interval {
            if last_archive.elapsed() >= interval {
                let archive_started = Instant::now();
                let unix_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs();
                let archive_path = run.archive_checkpoint_path(iteration, unix_secs)?;
                vs.save(&archive_path)?;
                println!("archived checkpoint: {}", archive_path.display());
                record["archive_checkpoint"] = archive_path.display().to_string().into();
                record["archive_checkpoint_secs"] = archive_started.elapsed().as_secs_f64().into();
                last_archive = Instant::now();
            }
        }

        run.log_metrics(record)?;
        iteration += 1;
    }
    Ok(())
}
