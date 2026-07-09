use crate::players::{AgentSpec, AlphaZeroAgent, HumanAgent, PlayerAgent};
use crate::proxy::{open_existing_run, resolve_model, run_dir, serve, GameKind, ServeConfig};
use anyhow::Result;
use clap::{Parser, Subcommand};
use engine_core::agent::{Agent, PolicyMode};
use engine_core::game::Game;
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
        GameKind::Chess => run_game::<ChessGame>(args),
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

fn run_game<G: Game>(args: PlayArgs) -> Result<()> {
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
        let action = if side == 0 {
            first.act_with_mode(&game, mode)
        } else {
            second.act_with_mode(&game, mode)
        };
        println!(
            "{} plays {}",
            side_name::<G>(side),
            game.format_action(action)
        );
        game.step(action);
        println!("{game}");
    }
    Ok(())
}

fn build_agent<G: Game>(
    spec: &AgentSpec,
    run_dir: &std::path::Path,
    device: Device,
    simulations: usize,
    wait_for_count: usize,
) -> Result<PlayerAgent> {
    match spec {
        AgentSpec::User => Ok(PlayerAgent::Human(HumanAgent)),
        AgentSpec::AlphaZero { model } => {
            let (_, cfg) = open_existing_run::<G>(run_dir)?;
            let weights = resolve_model(run_dir, model);
            Ok(PlayerAgent::AlphaZero(Box::new(AlphaZeroAgent::new(
                &cfg.net,
                &weights,
                device,
                simulations,
                wait_for_count,
                Duration::from_millis(1),
            )?)))
        }
    }
}

fn parse_agent(s: &str) -> Result<AgentSpec> {
    AgentSpec::parse(s)
}

fn side_name<G: Game>(side: usize) -> &'static str {
    if G::NAME == ChessGame::NAME {
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
