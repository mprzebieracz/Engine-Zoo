use crate::analysis::{Analysis, AnalyzeMode};
use crate::position::{ChessPosition, Connect4Position, PositionSpec};
use crate::proxy::{analyze_request, run_dir, AnalyzeRequest, GameKind};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tch::Device;

#[derive(Parser)]
#[command(about = "Developer evaluation tools for engine-zoo models")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Query policy + value for one position.
    Analyze(AnalyzeArgs),
    /// Run a JSONL puzzle/position suite against a model.
    Bench(BenchArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliAnalyzeMode {
    Net,
    Mcts,
}

impl From<CliAnalyzeMode> for AnalyzeMode {
    fn from(value: CliAnalyzeMode) -> Self {
        match value {
            CliAnalyzeMode::Net => AnalyzeMode::Net,
            CliAnalyzeMode::Mcts => AnalyzeMode::Mcts,
        }
    }
}

#[derive(Parser)]
struct AnalyzeArgs {
    #[arg(long, value_enum)]
    game: GameKind,
    #[arg(long)]
    run_dir: Option<PathBuf>,
    #[arg(long, default_value = "best")]
    model: String,
    #[arg(long, value_enum, default_value_t = CliAnalyzeMode::Net)]
    mode: CliAnalyzeMode,
    /// Chess FEN. If omitted, the start position is used.
    #[arg(long)]
    fen: Option<String>,
    /// Moves after the start position/FEN. Chess uses UCI; Connect4 uses columns.
    #[arg(long, value_delimiter = ',')]
    moves: Vec<String>,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    #[arg(long, default_value_t = 1)]
    wait_for_count: usize,
}

#[derive(Parser)]
struct BenchArgs {
    #[arg(long, value_enum)]
    game: GameKind,
    #[arg(long)]
    run_dir: Option<PathBuf>,
    #[arg(long, default_value = "best")]
    model: String,
    #[arg(long, value_enum, default_value_t = CliAnalyzeMode::Mcts)]
    mode: CliAnalyzeMode,
    /// JSONL file: {"position": {...}, "expected": ["move"]}
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    #[arg(long, default_value_t = 1)]
    wait_for_count: usize,
}

#[derive(Debug, Deserialize)]
struct BenchCase {
    position: PositionSpec,
    #[serde(default)]
    expected: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BenchResult {
    expected: Vec<String>,
    best_move: Option<String>,
    correct: bool,
    analysis: Analysis,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Analyze(args) => analyze_cmd(args),
        Command::Bench(args) => bench_cmd(args),
    }
}

fn analyze_cmd(args: AnalyzeArgs) -> Result<()> {
    let position = position_from_cli(args.game, args.fen, args.moves)?;
    let req = AnalyzeRequest {
        position,
        model: args.model,
        mode: Some(args.mode.into()),
        simulations: args.simulations,
        wait_for_count: args.wait_for_count,
    };
    let out = analyze_request(
        args.game,
        run_dir(args.game, args.run_dir),
        req,
        Device::cuda_if_available(),
    )?;
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn bench_cmd(args: BenchArgs) -> Result<()> {
    let run_dir = run_dir(args.game, args.run_dir);
    let device = Device::cuda_if_available();
    let mut rows = Vec::new();
    for (line_no, line) in fs::read_to_string(&args.suite)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let case: BenchCase = serde_json::from_str(line)
            .with_context(|| format!("parsing {}:{}", args.suite.display(), line_no + 1))?;
        let req = AnalyzeRequest {
            position: case.position,
            model: args.model.clone(),
            mode: Some(args.mode.into()),
            simulations: args.simulations,
            wait_for_count: args.wait_for_count,
        };
        let analysis = analyze_request(args.game, run_dir.clone(), req, device)?;
        let correct = analysis
            .best_move
            .as_ref()
            .is_some_and(|mv| case.expected.iter().any(|e| e == mv));
        rows.push(BenchResult {
            expected: case.expected,
            best_move: analysis.best_move.clone(),
            correct,
            analysis,
        });
    }

    let mut out: Box<dyn Write> = match args.output {
        Some(path) => Box::new(fs::File::create(path)?),
        None => Box::new(std::io::stdout()),
    };
    let correct = rows.iter().filter(|r| r.correct).count();
    writeln!(
        out,
        "{}",
        serde_json::to_string(&json!({
            "cases": rows.len(),
            "correct": correct,
            "accuracy": if rows.is_empty() { 0.0 } else { correct as f64 / rows.len() as f64 },
        }))?
    )?;
    for row in rows {
        writeln!(out, "{}", serde_json::to_string(&row)?)?;
    }
    Ok(())
}

fn position_from_cli(
    game: GameKind,
    fen: Option<String>,
    moves: Vec<String>,
) -> Result<PositionSpec> {
    Ok(match game {
        GameKind::Chess => PositionSpec::Chess(ChessPosition { fen, moves }),
        GameKind::Connect4 => {
            anyhow::ensure!(fen.is_none(), "connect4 positions use --moves, not --fen");
            let moves = moves
                .into_iter()
                .map(|m| {
                    m.parse::<u32>()
                        .with_context(|| format!("parsing move {m}"))
                })
                .collect::<Result<Vec<_>>>()?;
            PositionSpec::Connect4(Connect4Position { moves })
        }
    })
}
