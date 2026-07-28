use crate::proxy::{analyze_request, run_dir, AnalyzeRequest, GameKind};
use crate::visualization::{
    chess_board_for_position, render_bench_report, BenchReport, BenchReportRow,
};
use alphazero::{Analysis, AnalyzeMode};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use games::setup::{ChessSetup, Connect4Setup, GameSetup};
use serde::{Deserialize, Serialize};
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
    /// Remote server base URL. If set, query <server>/api/analyze instead of loading a local model.
    #[arg(long)]
    server: Option<String>,
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
    /// Remote server base URL. If set, query <server>/api/analyze instead of loading a local model.
    #[arg(long)]
    server: Option<String>,
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
    /// Optional standalone HTML report path.
    #[arg(long)]
    html: Option<PathBuf>,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    #[arg(long, default_value_t = 1)]
    wait_for_count: usize,
}

#[derive(Debug, Deserialize)]
struct BenchCase {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    category: Option<String>,
    position: GameSetup,
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

#[derive(Serialize)]
struct BenchSummary {
    cases: usize,
    correct: usize,
    accuracy: f64,
}

struct BenchRunner {
    game: GameKind,
    run_dir: PathBuf,
    device: Device,
    server: Option<String>,
}

impl BenchRunner {
    fn from_args(args: &BenchArgs) -> Self {
        Self {
            game: args.game,
            run_dir: run_dir(args.game, args.run_dir.clone()),
            device: Device::cuda_if_available(),
            server: args.server.clone(),
        }
    }

    fn analyze(&self, args: &BenchArgs, position: GameSetup) -> Result<Analysis> {
        let request = AnalyzeRequest {
            position,
            model: args.model.clone(),
            mode: Some(args.mode.into()),
            simulations: args.simulations,
            wait_for_count: args.wait_for_count,
        };

        match self.server.as_deref() {
            Some(server) => remote_analyze_request(server, &request),
            None => analyze_request(self.game, self.run_dir.clone(), request, self.device),
        }
    }
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
    let out = match args.server {
        Some(server) => remote_analyze_request(&server, &req)?,
        None => analyze_request(
            args.game,
            run_dir(args.game, args.run_dir),
            req,
            Device::cuda_if_available(),
        )?,
    };
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn bench_cmd(args: BenchArgs) -> Result<()> {
    let cases = read_bench_cases(&args.suite)?;
    let runner = BenchRunner::from_args(&args);

    let mut rows = Vec::new();
    let mut report_rows = Vec::new();

    for case in cases {
        let analysis = runner.analyze(&args, case.position.clone())?;
        let correct = analysis
            .best_move
            .as_ref()
            .is_some_and(|mv| case.expected.iter().any(|e| e == mv));
        let expected = case.expected.clone();
        let best_move = analysis.best_move.clone();

        if args.html.is_some() {
            report_rows.push(report_row(case, analysis.clone(), correct)?);
        }

        rows.push(BenchResult {
            expected,
            best_move,
            correct,
            analysis,
        });
    }

    write_bench_results(args.output.as_ref(), &rows)?;
    write_bench_html(args.html.as_ref(), &args, &report_rows)?;

    Ok(())
}

fn read_bench_cases(suite: &PathBuf) -> Result<Vec<BenchCase>> {
    fs::read_to_string(suite)?
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line_no, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("parsing {}:{}", suite.display(), line_no + 1))
        })
        .collect()
}

fn report_row(case: BenchCase, analysis: Analysis, correct: bool) -> Result<BenchReportRow> {
    let board = chess_board_for_position(&case.position)?;

    Ok(BenchReportRow {
        name: case.name,
        category: case.category,
        position: case.position,
        board,
        expected: case.expected,
        best_move: analysis.best_move.clone(),
        correct,
        analysis,
    })
}

fn write_bench_results(output: Option<&PathBuf>, rows: &[BenchResult]) -> Result<()> {
    let mut out: Box<dyn Write> = match output {
        Some(path) => Box::new(fs::File::create(path)?),
        None => Box::new(std::io::stdout()),
    };

    writeln!(out, "{}", serde_json::to_string(&bench_summary(rows))?)?;
    for row in rows {
        writeln!(out, "{}", serde_json::to_string(&row)?)?;
    }

    Ok(())
}

fn bench_summary(rows: &[BenchResult]) -> BenchSummary {
    let correct = rows.iter().filter(|row| row.correct).count();
    let accuracy = if rows.is_empty() {
        0.0
    }
    else {
        correct as f64 / rows.len() as f64
    };

    BenchSummary {
        cases: rows.len(),
        correct,
        accuracy,
    }
}

fn write_bench_html(
    html: Option<&PathBuf>,
    args: &BenchArgs,
    report_rows: &[BenchReportRow],
) -> Result<()> {
    let Some(path) = html
    else {
        return Ok(());
    };
    let report = BenchReport {
        title: "engine-zoo Puzzle Evaluation",
        game: args.game,
        model: &args.model,
        mode: match args.mode {
            CliAnalyzeMode::Net => "network",
            CliAnalyzeMode::Mcts => "mcts",
        },
        rows: report_rows,
    };

    fs::write(path, render_bench_report(&report))?;

    Ok(())
}

fn remote_analyze_request(server: &str, req: &AnalyzeRequest) -> Result<Analysis> {
    let url = format!("{}/api/analyze", server.trim_end_matches('/'));
    let client = reqwest::blocking::Client::new();
    let query = reqwest::Method::from_bytes(b"QUERY")?;
    let response = client
        .request(query, &url)
        .json(req)
        .send()
        .or_else(|_| client.post(&url).json(req).send())?;
    let response = if response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED {
        client.post(&url).json(req).send()?
    }
    else {
        response
    };
    let status = response.status();
    let body = response.text()?;
    anyhow::ensure!(
        status.is_success(),
        "remote analyze failed ({status}): {body}"
    );
    Ok(serde_json::from_str(&body)?)
}

fn position_from_cli(game: GameKind, fen: Option<String>, moves: Vec<String>) -> Result<GameSetup> {
    Ok(match game {
        GameKind::Chess => GameSetup::Chess(ChessSetup { fen, moves }),
        GameKind::Connect4 => {
            anyhow::ensure!(fen.is_none(), "connect4 positions use --moves, not --fen");
            let moves = moves
                .into_iter()
                .map(|m| {
                    m.parse::<u32>()
                        .with_context(|| format!("parsing move {m}"))
                })
                .collect::<Result<Vec<_>>>()?;
            GameSetup::Connect4(Connect4Setup { moves })
        }
    })
}
