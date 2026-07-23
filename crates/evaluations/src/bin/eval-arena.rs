use anyhow::Result;
use checkpoint_eval::fastchess::{FastchessCommand, Openings};
use checkpoint_eval::match_suite::run_match;
use checkpoint_eval::model_engine::{infer_run_dir, ModelEngine};
use checkpoint_eval::report::{EvaluationSpec, SearchSpec, SCHEMA_VERSION};
use clap::Parser;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    about = "Run a paired-opening Fastchess arena between independently configured chess checkpoints"
)]
struct Args {
    #[arg(long, default_value = "fastchess")]
    fastchess: PathBuf,
    #[arg(long, default_value = "target/release/engine-zoo-uci")]
    uci: PathBuf,
    #[arg(long)]
    candidate: PathBuf,
    #[arg(long)]
    baseline: PathBuf,
    /// Accepted for compatibility with the shared arena scripts; run directories
    /// are otherwise inferred from checkpoint locations.
    #[arg(long)]
    run_dir: Option<PathBuf>,
    /// Optional paired EPD opening set.
    #[arg(long)]
    openings: Option<PathBuf>,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    #[arg(long)]
    candidate_simulations: Option<usize>,
    #[arg(long)]
    baseline_simulations: Option<usize>,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long)]
    candidate_device: Option<String>,
    #[arg(long)]
    baseline_device: Option<String>,
    /// Total games; must be even because every game is paired with a colour reversal.
    #[arg(long, conflicts_with = "rounds")]
    games: Option<u32>,
    /// Number of colour-balanced pairs (equivalent to two games per round).
    #[arg(long, default_value_t = 2)]
    rounds: u32,
    #[arg(long, default_value_t = 1)]
    concurrency: u32,
    #[arg(long, default_value = "1000000+0")]
    tc: String,
    #[arg(long, default_value_t = 512)]
    max_moves: u32,
    /// Number of initial plies selected by each model's sampled policy.
    #[arg(long, default_value_t = 0)]
    opening_plies: u32,
}

fn validate_chess_run(run_dir: &Path) -> Result<()> {
    let (_, config, _) = alphazero::RunDir::open_or_create(run_dir, || {
        panic!("no experiment found at {}", run_dir.display())
    })?;
    anyhow::ensure!(
        config.model.game == alphazero::GameSpec::Chess,
        "{} is not a chess run",
        run_dir.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let rounds = args
        .games
        .map(|games| {
            anyhow::ensure!(
                games > 0 && games % 2 == 0,
                "--games must be a positive even number"
            );
            Ok(games / 2)
        })
        .transpose()?
        .unwrap_or(args.rounds);
    anyhow::ensure!(rounds > 0, "--rounds must be positive");
    anyhow::ensure!(args.simulations > 0, "--simulations must be positive");
    anyhow::ensure!(args.concurrency > 0, "--concurrency must be positive");
    anyhow::ensure!(args.max_moves > 0, "--max-moves must be positive");
    let candidate_run_dir = args
        .run_dir
        .clone()
        .unwrap_or_else(|| infer_run_dir(&args.candidate));
    let baseline_run_dir = args
        .run_dir
        .clone()
        .unwrap_or_else(|| infer_run_dir(&args.baseline));
    validate_chess_run(&candidate_run_dir)?;
    validate_chess_run(&baseline_run_dir)?;
    let candidate_simulations = args.candidate_simulations.unwrap_or(args.simulations);
    let baseline_simulations = args.baseline_simulations.unwrap_or(args.simulations);
    anyhow::ensure!(
        candidate_simulations > 0 && baseline_simulations > 0,
        "simulations must be positive"
    );
    let candidate_device = args.candidate_device.as_deref().unwrap_or(&args.device);
    let baseline_device = args.baseline_device.as_deref().unwrap_or(&args.device);
    let candidate = ModelEngine {
        name: "candidate",
        uci: &args.uci,
        run_dir: &candidate_run_dir,
        checkpoint: &args.candidate,
        simulations: candidate_simulations,
        device: candidate_device,
        opening_plies: Some(args.opening_plies),
    };
    let baseline = ModelEngine {
        name: "baseline",
        uci: &args.uci,
        run_dir: &baseline_run_dir,
        checkpoint: &args.baseline,
        simulations: baseline_simulations,
        device: baseline_device,
        opening_plies: Some(args.opening_plies),
    };
    let mut command =
        FastchessCommand::new(&args.fastchess, candidate.fastchess(), baseline.fastchess())
            // Fastchess requires a clock field; the default is practically unlimited.
            .time_control(&args.tc)
            .rounds(rounds)
            .concurrency(args.concurrency)
            .pgn_output(args.output_dir.join("games.pgn"))
            .arg("-maxmoves")
            .arg(args.max_moves.to_string());
    if let Some(openings) = &args.openings {
        command = command.openings(Openings::epd(openings));
    }
    let spec = EvaluationSpec {
        schema_version: SCHEMA_VERSION,
        id: format!(
            "arena-{}-vs-{}",
            args.candidate
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy(),
            args.baseline
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
        ),
        suite: "arena".into(),
        candidate: candidate.spec(),
        opponent: Some(baseline.spec()),
        search: SearchSpec {
            simulations: candidate_simulations,
            temperature: 0.0,
            dirichlet_noise: false,
            device: format!("candidate={candidate_device}, baseline={baseline_device}"),
        },
        opening_set: args.openings.map(|path| path.display().to_string()),
        seed: None,
        concurrency: args.concurrency as usize,
        fastchess_version: None,
    };
    let report = run_match(&args.output_dir, spec, command)?;
    println!(
        "arena: {}/{} ({:.1}%), {:+.1} local Elo",
        report.score.points(),
        report.score.games(),
        report.score_fraction * 100.0,
        report.smoothed_elo_delta
    );
    Ok(())
}
