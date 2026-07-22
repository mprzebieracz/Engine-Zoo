use anyhow::{Context, Result};
use checkpoint_eval::puzzle::{evaluate_checkpoint, read_puzzles, write_jsonl};
use clap::Parser;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use tch::Device;

#[derive(Parser, Debug)]
#[command(about = "Evaluate a chess checkpoint on a deterministic JSONL puzzle suite")]
struct Args {
    #[arg(long)]
    suite: PathBuf,
    #[arg(long)]
    run_dir: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 400)]
    simulations: usize,
    #[arg(long, default_value_t = 1)]
    leaf_batch_size: usize,
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn device(value: &str) -> Result<Device> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)),
        "auto" => Ok(Device::cuda_if_available()),
        other => anyhow::bail!("unknown --device {other}; use cpu, cuda, or auto"),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.simulations > 0, "--simulations must be positive");
    anyhow::ensure!(
        args.leaf_batch_size > 0,
        "--leaf-batch-size must be positive"
    );
    let puzzles = read_puzzles(BufReader::new(
        File::open(&args.suite).with_context(|| format!("opening {}", args.suite.display()))?,
    ))?;
    let (results, summary) = evaluate_checkpoint(
        &puzzles,
        &args.run_dir,
        &args.checkpoint,
        device(&args.device)?,
        alphazero::SearchConfig::Puct(alphazero::PuctConfig {
            common: alphazero::CommonSearchConfig {
                simulations: args.simulations,
                leaf_batch_size: args.leaf_batch_size,
            },
            root_noise: None,
            ..Default::default()
        }),
    )?;
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    write_jsonl(
        BufWriter::new(File::create(&args.output)?),
        &results,
        &summary,
    )?;
    println!(
        "evaluated {} puzzles: {}/{} correct ({:.2}%)",
        summary.total,
        summary.correct,
        summary.total,
        summary.accuracy * 100.0
    );
    Ok(())
}
