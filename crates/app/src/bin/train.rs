use alphazero::representation::{
    ChessAzRepresentation, ChessClassicRepresentation, Connect4AzRepresentation,
};
use alphazero::{
    Batcher, BatcherConfig, ChessHistoryLength, ChessSelfPlayWorkerFactory, ExperimentConfig,
    GameSpec, GenericSelfPlayWorkerFactory, ModelSpec, Network, ReplayBuffer, ReplayConfig, RunDir,
    SelfPlayConfig, SelfPlayCoordinator, TrainConfig, ValueHeadConfig, EXPERIMENT_FORMAT_VERSION,
};
use anyhow::Result;
use clap::{Parser, ValueEnum};
use games::{ChessPosition, Connect4};
use std::path::PathBuf;
use std::time::Duration;
use tch::{nn, Device};

#[derive(Clone, Copy, ValueEnum)]
enum GameKind {
    Connect4,
    Chess,
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceKind {
    Auto,
    Cuda,
    Cpu,
}

#[derive(Parser)]
#[command(about = "AlphaZero self-play and training")]
struct Args {
    #[arg(long, value_enum)]
    game: GameKind,
    #[arg(long)]
    run_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    #[arg(long)]
    forever: bool,
    #[arg(long, default_value_t = 100)]
    games: usize,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    #[arg(long, default_value_t = 1)]
    leaf_batch_size: usize,
    #[arg(long, default_value_t = 512)]
    max_moves: usize,
    #[arg(long, default_value_t = 500_000)]
    replay_capacity: usize,
    #[arg(long, default_value_t = 256)]
    batch_size: usize,
    #[arg(long, default_value_t = 80)]
    train_steps: usize,
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,
    #[arg(long, default_value_t = 1e-4)]
    weight_decay: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 4)]
    chess_history: usize,
    #[arg(long)]
    chess_classic: bool,
    #[arg(long, default_value_t = 12)]
    blocks: usize,
    #[arg(long, default_value_t = 128)]
    channels: i64,
    #[arg(long, value_enum, default_value_t = DeviceKind::Auto)]
    device: DeviceKind,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = args.run_dir.clone().unwrap_or_else(|| match args.game {
        GameKind::Connect4 => PathBuf::from("runs/connect4"),
        GameKind::Chess if args.chess_classic => PathBuf::from("runs/chess-classic"),
        GameKind::Chess => PathBuf::from(format!("runs/chess-h{}", args.chess_history)),
    });
    let (run, experiment, mut state) =
        RunDir::open_or_create(&root, || experiment_from_args(&args))?;
    let device = select_device(args.device)?;
    let mut vs = nn::VarStore::new(device);
    let net = Network::new(&vs.root(), &experiment.model)?;
    if let Some(checkpoint) = run.latest_checkpoint(&state) {
        vs.load(checkpoint)?;
    } else {
        run.write_latest(&mut state, |path| Ok(vs.save(path)?))?;
    }
    let batcher = Batcher::new_with_model(
        experiment.model.clone(),
        &run.latest_path(),
        device,
        BatcherConfig {
            preferred_batch_size: experiment.self_play.threads.max(1),
            max_batch_size: 256,
            max_wait: Duration::from_millis(2),
            max_queued_states: 4096,
        },
    )?;
    match experiment.model.game {
        GameSpec::Connect4 => run_connect4(
            &args,
            &run,
            &experiment,
            &mut state,
            &vs,
            &net,
            &batcher,
            device,
        ),
        GameSpec::Chess => run_chess(
            &args,
            &run,
            &experiment,
            &mut state,
            &vs,
            &net,
            &batcher,
            device,
        ),
    }
}

#[allow(clippy::too_many_arguments)] // Concrete model dispatch keeps the generic training loop type-safe.
fn run_connect4(
    args: &Args,
    run: &RunDir,
    experiment: &ExperimentConfig,
    state: &mut alphazero::RunState,
    vs: &nn::VarStore,
    net: &Network,
    batcher: &Batcher,
    device: Device,
) -> Result<()> {
    let replay =
        ReplayBuffer::<Connect4>::new(experiment.replay.capacity, experiment.model.action_size());
    let factory = GenericSelfPlayWorkerFactory::<Connect4, Connect4AzRepresentation>::new(
        batcher,
        experiment.self_play.clone(),
    )?;
    run_iterations(
        args,
        run,
        experiment,
        state,
        vs,
        net,
        batcher,
        device,
        &replay,
        &Connect4AzRepresentation,
        &factory,
    )
}

#[allow(clippy::too_many_arguments)] // Concrete model dispatch keeps the generic training loop type-safe.
fn run_chess(
    args: &Args,
    run: &RunDir,
    experiment: &ExperimentConfig,
    state: &mut alphazero::RunState,
    vs: &nn::VarStore,
    net: &Network,
    batcher: &Batcher,
    device: Device,
) -> Result<()> {
    if experiment.model.is_chess_classic() {
        return run_chess_classic(args, run, experiment, state, vs, net, batcher, device);
    }
    match experiment
        .model
        .chess_history()
        .expect("validated chess model")
    {
        ChessHistoryLength::One => {
            run_chess_history::<1>(args, run, experiment, state, vs, net, batcher, device)
        }
        ChessHistoryLength::Four => {
            run_chess_history::<4>(args, run, experiment, state, vs, net, batcher, device)
        }
        ChessHistoryLength::Eight => {
            run_chess_history::<8>(args, run, experiment, state, vs, net, batcher, device)
        }
    }
}

#[allow(clippy::too_many_arguments)] // Concrete model dispatch keeps the generic training loop type-safe.
fn run_chess_classic(
    args: &Args,
    run: &RunDir,
    experiment: &ExperimentConfig,
    state: &mut alphazero::RunState,
    vs: &nn::VarStore,
    net: &Network,
    batcher: &Batcher,
    device: Device,
) -> Result<()> {
    let replay = ReplayBuffer::new(experiment.replay.capacity, experiment.model.action_size());
    let factory = GenericSelfPlayWorkerFactory::<ChessPosition, ChessClassicRepresentation>::new(
        batcher,
        experiment.self_play.clone(),
    )?;
    run_iterations(
        args,
        run,
        experiment,
        state,
        vs,
        net,
        batcher,
        device,
        &replay,
        &ChessClassicRepresentation,
        &factory,
    )
}

#[allow(clippy::too_many_arguments)] // Concrete model dispatch keeps the generic training loop type-safe.
fn run_chess_history<const HISTORY: usize>(
    args: &Args,
    run: &RunDir,
    experiment: &ExperimentConfig,
    state: &mut alphazero::RunState,
    vs: &nn::VarStore,
    net: &Network,
    batcher: &Batcher,
    device: Device,
) -> Result<()> {
    let replay = ReplayBuffer::new(experiment.replay.capacity, experiment.model.action_size());
    let factory =
        ChessSelfPlayWorkerFactory::<HISTORY>::new(batcher, experiment.self_play.clone())?;
    run_iterations(
        args,
        run,
        experiment,
        state,
        vs,
        net,
        batcher,
        device,
        &replay,
        &ChessAzRepresentation::<HISTORY>,
        &factory,
    )
}

#[allow(clippy::too_many_arguments)] // Generic state, representation, model, and worker inputs meet here once.
fn run_iterations<S, R, F>(
    args: &Args,
    run: &RunDir,
    experiment: &ExperimentConfig,
    state: &mut alphazero::RunState,
    vs: &nn::VarStore,
    net: &Network,
    batcher: &Batcher,
    device: Device,
    replay: &ReplayBuffer<S>,
    representation: &R,
    factory: &F,
) -> Result<()>
where
    S: engine_core::GameState + Clone + Send + Sync + 'static,
    S::Move: Send + Sync,
    R: alphazero::AlphaZeroRepresentation<S>,
    F: alphazero::SelfPlayWorkerFactory<S>,
{
    let mut optimizer = alphazero::build_optimizer(vs, &experiment.training)?;
    let coordinator = SelfPlayCoordinator::new(experiment.self_play.clone(), experiment.seed)?;
    let mut completed = 0;
    while args.forever || completed < args.iterations {
        let stats = coordinator.run(factory, replay)?;
        let metrics = alphazero::train(
            net,
            &mut optimizer,
            replay,
            representation,
            device,
            &experiment.training,
        );
        state.iteration += 1;
        state.model_generation += 1;
        state.global_step += metrics
            .as_ref()
            .map_or(0, |metrics| metrics.train_steps as u64);
        state.replay_sample_count = replay.len();
        state.optimizer_moments_restored = false;
        run.write_latest(state, |path| Ok(vs.save(path)?))?;
        batcher
            .reload_weights(&run.latest_path())
            .map_err(anyhow::Error::msg)?;
        run.log_metrics(serde_json::json!({
            "iteration": state.iteration,
            "model_generation": state.model_generation,
            "games": stats.games,
            "moves": stats.moves,
            "replay_samples": replay.len(),
            "policy_loss": metrics.as_ref().map(|metrics| metrics.policy_loss),
            "value_loss": metrics.as_ref().map(|metrics| metrics.value_loss),
        }))?;
        completed += 1;
    }
    Ok(())
}

fn experiment_from_args(args: &Args) -> ExperimentConfig {
    let model = match args.game {
        GameKind::Connect4 => ModelSpec::connect4_basic(args.blocks, args.channels),
        GameKind::Chess if args.chess_classic => {
            ModelSpec::chess_classic(args.blocks, args.channels)
        }
        GameKind::Chess => ModelSpec::chess_se(
            history(args.chess_history),
            ValueHeadConfig::Wdl {
                hidden: args.channels,
            },
        ),
    };
    let mut self_play = SelfPlayConfig {
        num_games: args.games,
        threads: args.threads.unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, |threads| threads.get())
        }),
        max_moves: args.max_moves,
        ..Default::default()
    };
    if let search::SearchConfig::Puct(search) = &mut self_play.search {
        search.common.simulations = args.simulations;
        search.common.leaf_batch_size = args.leaf_batch_size;
    }
    self_play.budget_schedule = alphazero::SearchBudgetSchedule::fixed(&self_play.search);
    ExperimentConfig {
        format_version: EXPERIMENT_FORMAT_VERSION,
        model,
        self_play,
        replay: ReplayConfig {
            capacity: args.replay_capacity,
        },
        training: TrainConfig {
            batch_size: args.batch_size,
            train_steps: args.train_steps,
            lr: args.learning_rate,
            weight_decay: args.weight_decay,
            ..Default::default()
        },
        seed: args.seed,
    }
}

fn history(value: usize) -> ChessHistoryLength {
    match value {
        1 => ChessHistoryLength::One,
        4 => ChessHistoryLength::Four,
        8 => ChessHistoryLength::Eight,
        _ => panic!("--chess-history must be 1, 4, or 8"),
    }
}

fn select_device(kind: DeviceKind) -> Result<Device> {
    Ok(match kind {
        DeviceKind::Auto => Device::cuda_if_available(),
        DeviceKind::Cuda => {
            anyhow::ensure!(
                tch::Cuda::is_available(),
                "CUDA was requested but is unavailable"
            );
            Device::Cuda(0)
        }
        DeviceKind::Cpu => Device::Cpu,
    })
}
