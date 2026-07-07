use anyhow::Result;
use clap::{Parser, ValueEnum};
use engine_zoo::alphazero::{
    self_play, AlphaZeroNet, Batcher, Mcts, MctsConfig, MctsVariant, NetConfig, ReplayBuffer,
    RunConfig, RunDir, SelfPlayConfig, TrainConfig, train,
};
use engine_zoo::arena::{self, ArenaConfig};
use engine_zoo::game::Game;
use engine_zoo::games::{ChessGame, Connect4};
use serde_json::json;
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

#[derive(Parser)]
#[command(about = "AlphaZero self-play + training loop")]
struct Args {
    #[arg(long, value_enum)]
    game: GameKind,
    /// Run directory (checkpoints, metrics); defaults to runs/<game>.
    #[arg(long)]
    run_dir: Option<PathBuf>,

    #[arg(long, default_value_t = 10)]
    iterations: usize,
    #[arg(long, default_value_t = 100)]
    games: usize,
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    #[arg(long, default_value_t = 32)]
    mcts_batch: usize,
    #[arg(long, value_enum, default_value_t = SearchKind::Puct)]
    mcts_variant: SearchKind,
    /// Root actions considered by Gumbel MCTS before sequential halving.
    #[arg(long, default_value_t = 16)]
    gumbel_sampled_actions: usize,
    #[arg(long, default_value_t = 512)]
    max_moves: usize,

    #[arg(long, default_value_t = 52_500)]
    buffer: usize,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 4096)]
    minibatch_size: usize,
    #[arg(long, default_value_t = 20)]
    train_steps: usize,
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

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// How long the batcher waits for more states before running a partial batch.
const BATCH_TIMEOUT: Duration = Duration::from_millis(2);

fn main() -> Result<()> {
    let args = Args::parse();
    match args.game {
        GameKind::Connect4 => run::<Connect4>(args, 5, 64),
        GameKind::Chess => run::<ChessGame>(args, 10, 128),
    }
}

fn run<G: Game>(args: Args, default_blocks: i64, default_filters: i64) -> Result<()> {
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
        lr: args.lr,
        weight_decay: args.weight_decay,
    };
    let mut opt = engine_zoo::alphazero::build_optimizer(&vs, &train_cfg)?;
    let sp_cfg = SelfPlayConfig {
        num_games: args.games,
        threads: args.threads,
        max_moves: args.max_moves,
        mcts: MctsConfig {
            simulations: args.simulations,
            batch_size: args.mcts_batch,
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

    for iteration in 0..args.iterations {
        println!("=== iteration {iteration} ===");
        let self_play_started = Instant::now();
        {
            let wait_for = args.threads.min(4) * args.mcts_batch;
            let batcher = Batcher::new(
                &cfg.net,
                &run.best_path(),
                device,
                wait_for,
                BATCH_TIMEOUT,
            )?;
            self_play::<G>(&batcher, &replay, &sp_cfg);
        }
        let self_play_secs = self_play_started.elapsed().as_secs_f64();
        println!(
            "self-play: {} games in {self_play_secs:.1}s ({:.1} games/s)",
            args.games,
            args.games as f64 / self_play_secs
        );

        let train_started = Instant::now();
        let metrics = train(
            &net,
            &mut opt,
            &replay,
            device,
            &cfg.net,
            &train_cfg,
        );
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
                    batch_size: 8,
                    eps: 0.0, // no exploration noise in evaluation play
                    variant: MctsVariant::Puct,
                    ..Default::default()
                };
                let wait_for = mcts_cfg.batch_size;
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
                    next_ckpt += 1;
                }
                else {
                    println!(
                        "candidate rejected: {:.1}% < {:.1}%; self-play keeps the old best",
                        100.0 * winrate,
                        100.0 * args.gate_threshold
                    );
                }
            }
        }

        run.log_metrics(record)?;
    }
    Ok(())
}
