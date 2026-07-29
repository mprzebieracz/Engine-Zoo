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
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = load_config(&args.config)?;
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
    Ok(())
}
