mod support;

use self::support::{apply_opening, Stockfish};
use algorithms::alphazero::{Batcher, Mcts, MctsConfig, RunConfig};
use anyhow::{bail, Context, Result};
use clap::Parser;
use engine_core::game::Game;
use games::chess::notation;
use games::ChessGame;
use rand::{rngs::StdRng, SeedableRng};
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tch::Device;

#[derive(Parser)]
#[command(about = "Run a small, balanced Stockfish match for one chess checkpoint")]
pub struct Args {
    #[arg(long)]
    run_dir: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long, default_value = "stockfish")]
    stockfish: PathBuf,
    #[arg(long, default_value_t = 1400)]
    stockfish_elo: u32,
    #[arg(long, default_value_t = 4)]
    games: usize,
    /// Checkpoint from the prior evaluation interval for a direct relative match.
    #[arg(long)]
    baseline: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    baseline_games: usize,
    #[arg(long, default_value_t = 400)]
    simulations: usize,
    #[arg(long, default_value_t = 200)]
    stockfish_movetime_ms: u64,
    #[arg(long, default_value_t = 1)]
    stockfish_threads: u32,
    #[arg(long, default_value_t = 4)]
    opening_plies: usize,
    #[arg(long, default_value_t = 512)]
    max_moves: usize,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Serialize)]
struct ResultRecord {
    checkpoint: String,
    stockfish_elo: u32,
    games: usize,
    model_wins: usize,
    draws: usize,
    model_losses: usize,
    score: f64,
    score_pct: f64,
    rough_elo_delta: f64,
    simulations: usize,
    stockfish_movetime_ms: u64,
    opening_plies: usize,
    seed: u64,
    pgn_dir: String,
    baseline: Option<String>,
    baseline_games: Option<usize>,
    baseline_wins: Option<usize>,
    baseline_draws: Option<usize>,
    baseline_losses: Option<usize>,
    baseline_score_pct: Option<f64>,
    baseline_rough_elo_delta: Option<f64>,
    time: f64,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.games > 0 && args.games.is_multiple_of(2),
        "--games must be a positive even number"
    );
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        "auto" => Device::cuda_if_available(),
        other => bail!("unknown --device {other}; use cpu, cuda, or auto"),
    };
    let config: RunConfig =
        serde_json::from_str(&fs::read_to_string(args.run_dir.join("config.json"))?)?;
    anyhow::ensure!(
        config.game == "chess",
        "{} is not a chess run",
        args.run_dir.display()
    );
    let batcher = Batcher::new(
        &config.net,
        &args.checkpoint,
        device,
        1,
        Duration::from_millis(2),
    )?;
    let mut model = Mcts::new(
        batcher.client(),
        MctsConfig {
            simulations: args.simulations,
            eps: 0.0,
            ..Default::default()
        },
    );
    let mut stockfish =
        Stockfish::start(&args.stockfish, args.stockfish_elo, args.stockfish_threads)?;
    let name = args
        .checkpoint
        .file_stem()
        .context("checkpoint needs a filename")?
        .to_string_lossy()
        .to_string();
    let out_dir = args.run_dir.join("evaluations");
    let pgn_dir = out_dir.join("pgn").join(&name).join("stockfish");
    fs::create_dir_all(&pgn_dir)?;
    let seed = args.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut wins, mut draws, mut losses) = (0usize, 0usize, 0usize);
    for game_idx in 0..args.games {
        let model_white = game_idx.is_multiple_of(2);
        let mut game = ChessGame::default();
        let mut uci_moves = Vec::new();
        let mut san_moves = Vec::new();
        let _ = apply_opening(
            &mut game,
            Some((&mut uci_moves, &mut san_moves)),
            args.opening_plies,
            &mut rng,
        );
        let mut ply = uci_moves.len();
        while !game.is_terminal() && ply < args.max_moves {
            let model_turn = (ply.is_multiple_of(2)) == model_white;
            let action = if model_turn {
                model
                    .search_with_mode(&game, engine_core::agent::PolicyMode::Deterministic)
                    .best_action()
            }
            else {
                let mv = stockfish.best_move(&uci_moves, args.stockfish_movetime_ms)?;
                game.parse_move(&mv)
                    .with_context(|| format!("Stockfish returned illegal move {mv}"))?
            };
            san_moves.push(game.san_for_action(action));
            uci_moves.push(game.format_action(action));
            game.step(action);
            ply += 1;
        }
        let result = if !game.is_terminal() || game.reward() == 0.0 {
            draws += 1;
            "1/2-1/2"
        }
        else if (ply - 1).is_multiple_of(2) == model_white {
            wins += 1;
            if model_white {
                "1-0"
            }
            else {
                "0-1"
            }
        }
        else {
            losses += 1;
            if model_white {
                "0-1"
            }
            else {
                "1-0"
            }
        };
        let movetext = notation::movetext(&san_moves, 0, result);
        let round = (game_idx + 1).to_string();
        let stockfish_elo = args.stockfish_elo.to_string();
        let white = if model_white { &name } else { "Stockfish" };
        let black = if model_white { "Stockfish" } else { &name };
        let pgn = notation::pgn(
            &[
                ("Event", "engine-zoo checkpoint evaluation"),
                ("Round", &round),
                ("White", white),
                ("Black", black),
                ("Result", result),
                ("StockfishElo", &stockfish_elo),
            ],
            &movetext,
        );
        fs::write(pgn_dir.join(format!("game_{:02}.pgn", game_idx + 1)), pgn)?;
    }
    let score = wins as f64 + draws as f64 * 0.5;
    let smoothed = (score + 0.5) / (args.games as f64 + 1.0);
    let baseline_result = if let Some(path) = args.baseline.as_ref() {
        anyhow::ensure!(
            args.baseline_games > 0 && args.baseline_games.is_multiple_of(2),
            "--baseline-games must be a positive even number"
        );
        let baseline_batcher =
            Batcher::new(&config.net, path, device, 1, Duration::from_millis(2))?;
        let mut baseline = Mcts::new(
            baseline_batcher.client(),
            MctsConfig {
                simulations: args.simulations,
                eps: 0.0,
                ..Default::default()
            },
        );
        let (mut baseline_wins, mut baseline_draws, mut baseline_losses) = (0usize, 0usize, 0usize);
        for game_idx in 0..args.baseline_games {
            let model_white = game_idx.is_multiple_of(2);
            let mut game = ChessGame::default();
            let mut ply = apply_opening(&mut game, None, args.opening_plies, &mut rng);
            while !game.is_terminal() && ply < args.max_moves {
                let action = if ply.is_multiple_of(2) == model_white {
                    model
                        .search_with_mode(&game, engine_core::agent::PolicyMode::Deterministic)
                        .best_action()
                }
                else {
                    baseline
                        .search_with_mode(&game, engine_core::agent::PolicyMode::Deterministic)
                        .best_action()
                };
                game.step(action);
                ply += 1;
            }
            if !game.is_terminal() || game.reward() == 0.0 {
                baseline_draws += 1;
            }
            else if (ply - 1).is_multiple_of(2) == model_white {
                baseline_wins += 1;
            }
            else {
                baseline_losses += 1;
            }
        }
        let baseline_score = baseline_wins as f64 + baseline_draws as f64 * 0.5;
        let baseline_smoothed = (baseline_score + 0.5) / (args.baseline_games as f64 + 1.0);
        Some((
            path.display().to_string(),
            baseline_wins,
            baseline_draws,
            baseline_losses,
            baseline_score * 100.0 / args.baseline_games as f64,
            400.0 * (baseline_smoothed / (1.0 - baseline_smoothed)).log10(),
        ))
    }
    else {
        None
    };
    let record = ResultRecord {
        checkpoint: args.checkpoint.display().to_string(),
        stockfish_elo: args.stockfish_elo,
        games: args.games,
        model_wins: wins,
        draws,
        model_losses: losses,
        score,
        score_pct: score * 100.0 / args.games as f64,
        rough_elo_delta: 400.0 * (smoothed / (1.0 - smoothed)).log10(),
        simulations: args.simulations,
        stockfish_movetime_ms: args.stockfish_movetime_ms,
        opening_plies: args.opening_plies,
        seed,
        pgn_dir: pgn_dir.display().to_string(),
        baseline: baseline_result.as_ref().map(|x| x.0.clone()),
        baseline_games: baseline_result.as_ref().map(|_| args.baseline_games),
        baseline_wins: baseline_result.as_ref().map(|x| x.1),
        baseline_draws: baseline_result.as_ref().map(|x| x.2),
        baseline_losses: baseline_result.as_ref().map(|x| x.3),
        baseline_score_pct: baseline_result.as_ref().map(|x| x.4),
        baseline_rough_elo_delta: baseline_result.as_ref().map(|x| x.5),
        time: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
    };
    let line = serde_json::to_string(&record)?;
    fs::create_dir_all(&out_dir)?;
    let results_path = out_dir.join("ratings.jsonl");
    let mut results = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&results_path)?;
    writeln!(results, "{line}")?;
    fs::write(out_dir.join(format!("{name}.json")), format!("{line}\n"))?;
    println!("evaluation: {line}");
    Ok(())
}
