use crate::players::{AgentSpec, AlphaZeroAgent, HumanAgent, PlayerAgent};
use crate::proxy::{open_existing_run, resolve_model, run_dir, serve, GameKind, ServeConfig};
use alphazero::GameSpec;
use anyhow::Result;
use clap::{Parser, Subcommand};
use engine_core::agent::{Agent, PolicyMode};
use games::{ChessGame, Connect4};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use tch::Device;

#[derive(Parser)]
#[command(about = "Play and watch engine-zoo games")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Play or watch a game between two agents.
    Play(PlayArgs),
    /// Serve the session API over HTTP for local or remote clients.
    Serve(ServeArgs),
}

#[derive(Parser)]
struct PlayArgs {
    #[arg(long, value_enum)]
    game: GameKind,
    #[arg(long)]
    run_dir: Option<PathBuf>,
    /// First player: White in chess, X in connect4. Examples: user, alphazero, alphazero:ckpt_0010.safetensors.
    #[arg(long, default_value = "user", value_parser = parse_agent)]
    first: AgentSpec,
    /// Second player: Black in chess, O in connect4.
    #[arg(long, default_value = "alphazero", value_parser = parse_agent)]
    second: AgentSpec,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    #[arg(long, default_value_t = 1)]
    wait_for_count: usize,
    #[arg(long, default_value_t = 512)]
    max_moves: usize,
    /// Opening plies sampled from engine policies instead of argmax.
    #[arg(long, default_value_t = 0)]
    opening_moves: usize,
}

#[derive(Parser)]
struct ServeArgs {
    #[arg(long, value_enum)]
    game: GameKind,
    #[arg(long)]
    run_dir: Option<PathBuf>,
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
}

#[derive(Clone, Copy, Debug)]
struct PlayConfig {
    max_moves: usize,
    opening_moves: usize,
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Play(args) => play_cmd(args),
        Command::Serve(args) => serve_cmd(args).await,
    }
}

fn play_cmd(args: PlayArgs) -> Result<()> {
    match args.game {
        GameKind::Connect4 => run_game::<Connect4>(args),
        GameKind::Chess => anyhow::bail!(
            "interactive chess uses a history-aware representation; use UCI or the session API"
        ),
    }
}

async fn serve_cmd(args: ServeArgs) -> Result<()> {
    serve(ServeConfig {
        game: args.game,
        run_dir: run_dir(args.game, args.run_dir),
        bind: args.bind,
    })
    .await
}

fn run_game<G: crate::players::InteractiveGame>(args: PlayArgs) -> Result<()> {
    let root = run_dir(args.game, args.run_dir);
    let device = Device::cuda_if_available();
    let mut first = build_agent::<G>(
        &args.first,
        &root,
        device,
        args.simulations,
        args.wait_for_count,
    )?;
    let mut second = build_agent::<G>(
        &args.second,
        &root,
        device,
        args.simulations,
        args.wait_for_count,
    )?;
    let mut game = G::default();
    let cfg = PlayConfig {
        max_moves: args.max_moves,
        opening_moves: args.opening_moves,
    };

    println!("{game}");
    for ply in 0..cfg.max_moves {
        if game.is_terminal() {
            break;
        }
        let side = ply % 2;
        let mode = if ply < cfg.opening_moves {
            PolicyMode::Explore
        } else {
            PolicyMode::Deterministic
        };
        let mv = if side == 0 {
            first.select_move(&game, mode)
        } else {
            second.select_move(&game, mode)
        };
        println!(
            "{} plays {}",
            side_name::<G>(side),
            game.format_native_move(mv)
        );
        game.play(mv);
        println!("{game}");
    }
    Ok(())
}

fn build_agent<G: crate::players::InteractiveGame>(
    spec: &AgentSpec,
    run_dir: &std::path::Path,
    device: Device,
    simulations: usize,
    wait_for_count: usize,
) -> Result<PlayerAgent<G>> {
    match spec {
        AgentSpec::User => Ok(PlayerAgent::Human(HumanAgent)),
        AgentSpec::AlphaZero { model } => {
            let (_, cfg) = open_existing_run(run_dir, interactive_game_name::<G>())?;
            let weights = resolve_model(run_dir, model);
            let compatible = matches!(
                (&cfg.model.game, interactive_game_name::<G>()),
                (GameSpec::Connect4, "connect4")
            );
            anyhow::ensure!(
                compatible,
                "interactive play only supports Connect4; use UCI or the session API for Chess"
            );
            Ok(PlayerAgent::AlphaZero(Box::new(AlphaZeroAgent::<G>::new(
                &cfg.model,
                &weights,
                device,
                simulations,
                wait_for_count,
                Duration::from_millis(1),
            )?)))
        }
    }
}

fn interactive_game_name<G: crate::players::InteractiveGame>() -> &'static str {
    if std::any::TypeId::of::<G>() == std::any::TypeId::of::<ChessGame>() {
        "chess"
    } else {
        "connect4"
    }
}

fn parse_agent(s: &str) -> Result<AgentSpec> {
    AgentSpec::parse(s)
}

fn side_name<G: crate::players::InteractiveGame>(side: usize) -> &'static str {
    if std::any::TypeId::of::<G>() == std::any::TypeId::of::<ChessGame>() {
        if side == 0 {
            "white"
        } else {
            "black"
        }
    } else if side == 0 {
        "first"
    } else {
        "second"
    }
}
