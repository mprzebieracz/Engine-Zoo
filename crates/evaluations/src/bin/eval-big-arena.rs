use anyhow::Result;
use checkpoint_eval::big_arena::{load_config, run, ArenaSummary};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Run a candidate checkpoint against every opponent in a TOML model roster")]
struct Args {
    /// TOML file describing the candidate, opponents, and arena settings.
    #[arg(long, default_value = "configs/arena.toml")]
    config: PathBuf,
    /// Override the candidate checkpoint path.
    #[arg(long)]
    candidate_path: Option<PathBuf>,
    /// Override the candidate display name.
    #[arg(long)]
    candidate_name: Option<String>,
    /// Override the candidate architecture label.
    #[arg(long)]
    candidate_architecture: Option<String>,
    /// Override the candidate simulations count.
    #[arg(long)]
    candidate_simulations: Option<usize>,
    /// Override the candidate's TensorRT module path.
    #[arg(long)]
    candidate_tensor_rt_module: Option<PathBuf>,
    /// Override the total games played per opponent pair.
    #[arg(long)]
    games: Option<u32>,
    /// Override the arena output directory.
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut config = load_config(&args.config)?;

    if let Some(path) = args.candidate_path {
        config.candidate.path = path;
    }
    if let Some(name) = args.candidate_name {
        config.candidate.name = name;
    }
    if let Some(architecture) = args.candidate_architecture {
        config.candidate.architecture = architecture;
    }
    if let Some(simulations) = args.candidate_simulations {
        config.candidate.simulations = Some(simulations);
    }
    if let Some(module) = args.candidate_tensor_rt_module {
        config.candidate.tensor_rt_module = Some(module);
    }
    if let Some(games) = args.games {
        config.settings.games = games;
    }
    if let Some(output_dir) = args.output_dir {
        config.settings.output_dir = output_dir;
    }

    let summary: ArenaSummary = run(&config)?;
    println!(
        "arena complete: {} opponents, {} games, {:.1}% score, {:+.1} Elo",
        summary.opponents.len(),
        summary.total_games,
        summary.score_fraction * 100.0,
        summary.elo_delta
    );
    println!(
        "results: {}/arena.json",
        config.settings.output_dir.display()
    );
    println!(
        "report:  {}/REPORT.md",
        config.settings.output_dir.display()
    );
    Ok(())
}
