use alphazero::representation::{
    ChessAzRepresentation, ChessClassicRepresentation, Connect4AzRepresentation,
};
use alphazero::{
    Batcher, ChessHistoryLength, ChessSelfPlayWorkerFactory, ExperimentConfig, GameSpec,
    GenericSelfPlayWorkerFactory, Network, ReplayBuffer, RunDir, SelfPlayCoordinator,
    SelfPlayEpoch,
};
use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use games::{ChessPosition, Connect4};
use std::path::PathBuf;
use tch::{nn, Device};

#[derive(Clone, Copy, ValueEnum)]
enum DeviceKind {
    Auto,
    Cuda,
    Cpu,
}

#[derive(Parser)]
#[command(about = "AlphaZero self-play and training")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate an immutable experiment TOML and initialize an empty run.
    Init {
        #[arg(long)]
        experiment: PathBuf,
        #[arg(long)]
        run_dir: PathBuf,
    },
    /// Resume an initialized run. Runtime flags do not change its experiment.
    Run(RunArgs),
    /// Print the immutable experiment and current mutable run state.
    Inspect {
        #[arg(long)]
        run_dir: PathBuf,
    },
}

#[derive(Parser)]
struct RunArgs {
    #[arg(long)]
    run_dir: PathBuf,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    #[arg(long)]
    forever: bool,
    #[arg(long, value_enum, default_value_t = DeviceKind::Auto)]
    device: DeviceKind,
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Init {
            experiment,
            run_dir,
        } => {
            let config = ExperimentConfig::read_toml(&experiment)?;
            RunDir::initialize(&run_dir, config)?;
            println!("initialized {}", run_dir.display());
            Ok(())
        }
        Command::Inspect { run_dir } => {
            let (_, experiment, state) = RunDir::open(&run_dir)?;
            println!("{}", experiment.to_toml()?);
            println!("state:");
            println!("{}", serde_json::to_string_pretty(&state)?);
            Ok(())
        }
        Command::Run(args) => run(args),
    }
}

fn run(args: RunArgs) -> Result<()> {
    let (run, experiment, mut state) = RunDir::open(&args.run_dir)?;
    let device = select_device(args.device)?;
    let mut vs = nn::VarStore::new(device);
    let net = Network::new(&vs.root(), &experiment.model)?;
    if let Some(checkpoint) = run.latest_checkpoint(&state) {
        vs.load(checkpoint)?;
        state.mark_weights_only_resume();
        run.write_state(&state)?;
        eprintln!(
            "WARNING: resumed checkpoint weights only; replay and optimizer moments were reset."
        );
        run.log_metrics(serde_json::json!({
            "event": "resume",
            "resume_kind": "weights-only",
            "replay_restored": false,
            "optimizer_moments_restored": false,
        }))?;
    } else {
        run.write_latest(&mut state, |path| Ok(vs.save(path)?))?;
    }
    let batcher = Batcher::new_with_model_precision(
        experiment.model.clone(),
        &run.latest_path(),
        device,
        experiment.inference.batcher_config(),
        experiment.inference.precision,
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
    args: &RunArgs,
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
    args: &RunArgs,
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
    args: &RunArgs,
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
    args: &RunArgs,
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
    args: &RunArgs,
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
        let epoch = SelfPlayEpoch {
            model_generation: state.model_generation,
            first_game_id: state.total_games_generated,
        };
        let stats = coordinator.run(factory, replay, epoch)?;
        state.total_games_generated += stats.games as u64;
        let metrics = alphazero::train(
            net,
            &mut optimizer,
            replay,
            representation,
            device,
            &experiment.training,
            experiment.seed,
            state.global_step,
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
            "total_games_generated": state.total_games_generated,
            "games": stats.games,
            "moves": stats.moves,
            "replay_samples": replay.len(),
            "policy_loss": metrics.as_ref().map(|metrics| metrics.policy_loss),
            "value_loss": metrics.as_ref().map(|metrics| metrics.value_loss),
            "learning_rate": metrics.as_ref().map(|metrics| metrics.learning_rate),
            "training": metrics,
            "batcher": batcher.stats(),
        }))?;
        completed += 1;
    }
    Ok(())
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
