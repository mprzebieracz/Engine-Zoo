mod batcher;
mod cli;
mod device;
mod environment;
mod harness;
mod inference;
mod iteration;
mod raw_inference;
mod replay;
mod report;
mod representation;
mod search;
mod self_play;
mod training;
mod trt_selfplay;

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let command = cli::Command::parse();
    cli::run(command)
}
