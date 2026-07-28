use crate::{
    batcher, device, environment, harness, inference, replay, representation, search, self_play,
    training,
};
use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "engine-bench",
    about = "Reproducible engine-zoo benchmark runner"
)]
pub struct Command {
    #[command(subcommand)]
    command: BenchmarkCommand,
}

#[derive(Subcommand)]
enum BenchmarkCommand {
    Environment(OutputArgs),
    Search(SearchArgs),
    Representation(OutputArgs),
    Inference(OutputArgs),
    Batcher(OutputArgs),
    #[command(name = "self-play")]
    SelfPlay(OutputArgs),
    Training(OutputArgs),
    #[command(name = "end-to-end")]
    EndToEnd(OutputArgs),
    Suite(OutputArgs),
}

#[derive(Args, Clone)]
pub struct OutputArgs {
    /// Optional TOML workload configuration recorded verbatim in the report.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Warmup iterations, or a duration such as `2s` or `500ms`.
    #[arg(long, default_value = "1")]
    warmup: String,

    #[arg(long, default_value_t = 10)]
    samples: usize,

    /// Execution device. CUDA is the default; use `auto` to permit CPU fallback.
    #[arg(long, value_enum, default_value_t = device::DeviceKind::Cuda)]
    device: device::DeviceKind,

    /// Network precision for CUDA inference fixtures.
    #[arg(long, value_enum, default_value_t = device::Precision::Fp32)]
    precision: device::Precision,

    /// Write the complete machine-readable JSON report to this path.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Also print a concise human-readable summary.
    #[arg(long)]
    human: bool,
}

#[derive(Args)]
struct SearchArgs {
    #[command(flatten)]
    output: OutputArgs,

    #[arg(long, value_enum, default_value_t = SearchAlgorithm::Puct)]
    algorithm: SearchAlgorithm,

    #[arg(long, default_value_t = 128)]
    simulations: usize,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum SearchAlgorithm {
    Puct,
    RootGumbelPuct,
    FullGumbel,
}

pub fn run(command: Command) -> Result<()> {
    match command.command {
        BenchmarkCommand::Environment(args) => write(environment::report()?, &args),
        BenchmarkCommand::Search(args) => {
            let config = config_value(&args.output)?;
            let report = match args.algorithm {
                SearchAlgorithm::Puct => search::puct(
                    args.simulations,
                    &args.output.warmup,
                    args.output.samples,
                    config,
                )?,
                SearchAlgorithm::RootGumbelPuct => search::root_gumbel_puct(
                    args.simulations,
                    &args.output.warmup,
                    args.output.samples,
                    config,
                )?,
                SearchAlgorithm::FullGumbel => search::full_gumbel(
                    args.simulations,
                    &args.output.warmup,
                    args.output.samples,
                    config,
                )?,
            };
            write(report, &args.output)
        }
        BenchmarkCommand::Representation(args) => {
            let report = representation::connect4(&args.warmup, args.samples, config_value(&args)?);
            write(report, &args)
        }
        BenchmarkCommand::Suite(args) => run_suite(&args),
        BenchmarkCommand::Inference(args) => write(
            inference::connect4(
                &args.warmup,
                args.samples,
                config_value(&args)?,
                benchmark_device(&args)?,
                args.precision,
            ),
            &args,
        ),
        BenchmarkCommand::Batcher(args) => write(
            batcher::connect4(
                &args.warmup,
                args.samples,
                config_value(&args)?,
                benchmark_device(&args)?,
                args.precision,
            )?,
            &args,
        ),
        BenchmarkCommand::SelfPlay(args) => write(
            self_play::connect4(
                &args.warmup,
                args.samples,
                config_value(&args)?,
                benchmark_device(&args)?,
                args.precision,
            )?,
            &args,
        ),
        BenchmarkCommand::Training(args) => write(
            training::connect4(
                &args.warmup,
                args.samples,
                config_value(&args)?,
                benchmark_device(&args)?,
                args.precision,
            )?,
            &args,
        ),
        BenchmarkCommand::EndToEnd(args) => write(end_to_end(&args)?, &args),
    }
}

fn run_suite(args: &OutputArgs) -> Result<()> {
    let config = config_value(args)?;
    let reports = vec![
        search::puct(128, &args.warmup, args.samples, config.clone())?,
        search::root_gumbel_puct(128, &args.warmup, args.samples, config.clone())?,
        search::full_gumbel(128, &args.warmup, args.samples, config.clone())?,
        representation::connect4(&args.warmup, args.samples, config),
        inference::connect4(
            &args.warmup,
            args.samples,
            json!({}),
            benchmark_device(args)?,
            args.precision,
        ),
        batcher::connect4(
            &args.warmup,
            args.samples,
            json!({}),
            benchmark_device(args)?,
            args.precision,
        )?,
        replay::connect4(&args.warmup, args.samples, json!({})),
        self_play::connect4(
            &args.warmup,
            args.samples,
            json!({}),
            benchmark_device(args)?,
            args.precision,
        )?,
        training::connect4(
            &args.warmup,
            args.samples,
            json!({}),
            benchmark_device(args)?,
            args.precision,
        )?,
        end_to_end(args)?,
    ];
    let report = harness::combine("suite", reports)?;
    write(report, args)
}

fn end_to_end(args: &OutputArgs) -> Result<crate::report::BenchmarkReport> {
    let config = config_value(args)?;
    let reports = vec![
        self_play::connect4(
            &args.warmup,
            args.samples,
            config.clone(),
            benchmark_device(args)?,
            args.precision,
        )?,
        training::connect4(
            &args.warmup,
            args.samples,
            config,
            benchmark_device(args)?,
            args.precision,
        )?,
    ];
    harness::combine("end-to-end.connect4", reports)
}

fn benchmark_device(args: &OutputArgs) -> Result<tch::Device> {
    let selected = device::select(args.device)?;
    device::validate_precision(args.precision, selected)?;
    Ok(selected)
}

fn config_value(args: &OutputArgs) -> Result<Value> {
    let Some(path) = &args.config
    else {
        return Ok(json!({}));
    };
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read benchmark config {}", path.display()))?;
    Ok(json!({ "path": path, "toml": contents }))
}

fn write(report: crate::report::BenchmarkReport, args: &OutputArgs) -> Result<()> {
    let json = serde_json::to_string_pretty(&report)?;
    println!("{json}");

    if args.human {
        print_human(&report);
    }

    if let Some(path) = &args.output {
        std::fs::write(path, format!("{json}\n"))
            .with_context(|| format!("failed to write benchmark report {}", path.display()))?;
    }

    Ok(())
}

fn print_human(report: &crate::report::BenchmarkReport) {
    if report.reports.is_empty() {
        println!(
            "{}: median {:.3} ms",
            report.benchmark,
            report.summary.median_ns as f64 / 1_000_000.0
        );
        return;
    }

    println!("{}:", report.benchmark);
    for child in &report.reports {
        print_human(child);
    }
}
