use anyhow::Result;
use checkpoint_eval::fastchess::{Engine, FastchessCommand, Openings};
use checkpoint_eval::match_suite::run_match;
use checkpoint_eval::model_engine::ModelEngine;
use checkpoint_eval::report::{EngineSpec, EvaluationSpec, SearchSpec, SCHEMA_VERSION};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Run a paired-opening Fastchess match between a checkpoint and Stockfish")]
struct Args {
    #[arg(long, default_value = "fastchess")]
    fastchess: PathBuf,
    #[arg(long, default_value = "target/release/engine-zoo-uci")]
    uci: PathBuf,
    #[arg(long)]
    stockfish: PathBuf,
    #[arg(long)]
    run_dir: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    openings: PathBuf,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value = "60+0.6")]
    tc: String,
    #[arg(long, default_value_t = 2)]
    rounds: u32,
    #[arg(long, default_value_t = 1)]
    concurrency: u32,
    #[arg(long, default_value_t = 1)]
    stockfish_threads: usize,
    #[arg(long, conflicts_with = "stockfish_elo")]
    stockfish_nodes: Option<usize>,
    #[arg(long, conflicts_with = "stockfish_nodes")]
    stockfish_elo: Option<u32>,
    #[arg(long, default_value_t = 512)]
    max_moves: u32,
}

fn validate(args: &Args) -> Result<()> {
    anyhow::ensure!(args.rounds > 0, "--rounds must be positive");
    anyhow::ensure!(args.simulations > 0, "--simulations must be positive");
    anyhow::ensure!(args.concurrency > 0, "--concurrency must be positive");
    anyhow::ensure!(
        args.stockfish_threads > 0,
        "--stockfish-threads must be positive"
    );
    anyhow::ensure!(args.max_moves > 0, "--max-moves must be positive");
    if let Some(nodes) = args.stockfish_nodes {
        anyhow::ensure!(nodes > 0, "--stockfish-nodes must be positive");
    }

    Ok(())
}

fn stockfish_opponent(args: &Args) -> (Engine, EngineSpec) {
    let mut engine = Engine::new(&args.stockfish, "stockfish")
        .option("Threads", args.stockfish_threads.to_string());
    let mut options = vec![("Threads".into(), args.stockfish_threads.to_string())];

    let name = if let Some(elo) = args.stockfish_elo {
        engine = engine
            .option("UCI_LimitStrength", "true")
            .option("UCI_Elo", elo.to_string());
        options.extend([
            ("UCI_LimitStrength".into(), "true".into()),
            ("UCI_Elo".into(), elo.to_string()),
        ]);

        format!("stockfish-elo-{elo}")
    }
    else {
        let nodes = args.stockfish_nodes.unwrap_or(3_000);
        engine = engine.nodes(nodes);

        format!("stockfish-nodes-{nodes}")
    };

    let spec = EngineSpec {
        name,
        command: args.stockfish.display().to_string(),
        args: vec![],
        checkpoint: None,
        options,
    };

    (engine, spec)
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate(&args)?;

    let candidate = ModelEngine {
        name: "candidate",
        uci: &args.uci,
        run_dir: &args.run_dir,
        checkpoint: &args.candidate,
        simulations: args.simulations,
        device: &args.device,
        opening_plies: None,
    };

    let (stockfish, opponent) = stockfish_opponent(&args);

    let command = FastchessCommand::new(&args.fastchess, candidate.fastchess(), stockfish)
        .time_control(&args.tc)
        .rounds(args.rounds)
        .concurrency(args.concurrency)
        .openings(Openings::epd(&args.openings))
        .pgn_output(args.output_dir.join("games.pgn"))
        .arg("-maxmoves")
        .arg(args.max_moves.to_string());
    let spec = EvaluationSpec {
        schema_version: SCHEMA_VERSION,
        id: format!(
            "stockfish-{}-vs-{}",
            args.candidate
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy(),
            opponent.name
        ),
        suite: "stockfish".into(),
        candidate: candidate.spec(),
        opponent: Some(opponent),
        search: SearchSpec {
            simulations: args.simulations,
            temperature: 0.0,
            dirichlet_noise: false,
            device: args.device,
        },
        opening_set: Some(args.openings.display().to_string()),
        seed: None,
        concurrency: args.concurrency as usize,
        fastchess_version: None,
    };
    let report = run_match(&args.output_dir, spec, command)?;
    println!(
        "stockfish: {}/{} ({:.1}%), {:+.1} local Elo",
        report.score.points(),
        report.score.games(),
        report.score_fraction * 100.0,
        report.smoothed_elo_delta
    );
    Ok(())
}
