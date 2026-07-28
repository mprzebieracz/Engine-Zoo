use crate::{
    batcher, device, environment, harness, inference, iteration, replay, representation, search,
    self_play, training,
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
    /// Run a complete self-play, training, checkpoint, and inference-reload iteration.
    Iteration(IterationArgs),
    #[command(name = "end-to-end")]
    EndToEnd(OutputArgs),
    Suite(OutputArgs),
}

#[derive(Args, Clone)]
pub(crate) struct IterationArgs {
    /// Immutable experiment TOML used to construct each measured training run.
    #[arg(
        long,
        default_value = "benchmarks/configs/full-iteration-cuda-200.toml"
    )]
    pub(crate) experiment: PathBuf,

    /// Override the configured number of self-play games.
    #[arg(long)]
    pub(crate) games: Option<usize>,

    /// Override the configured training steps per iteration.
    #[arg(long)]
    pub(crate) train_steps: Option<usize>,

    /// Override the configured training batch size.
    #[arg(long)]
    pub(crate) batch_size: Option<usize>,

    /// Override the configured training micro-batch size.
    #[arg(long)]
    pub(crate) micro_batch_size: Option<usize>,

    /// Override the deterministic experiment seed.
    #[arg(long)]
    pub(crate) seed: Option<u64>,

    /// Execution device. CUDA is the default; use `auto` to permit CPU fallback.
    #[arg(long, value_enum, default_value_t = device::DeviceKind::Cuda)]
    pub(crate) device: device::DeviceKind,

    /// Override the inference precision stored in the experiment.
    #[arg(long, value_enum)]
    pub(crate) precision: Option<device::Precision>,

    /// Full iterations to run before recording samples. Defaults to zero because a warmup changes replay contents.
    #[arg(long, default_value_t = 0)]
    pub(crate) warmup_iterations: usize,

    /// Number of independent, fresh full iterations to record.
    #[arg(long, default_value_t = 1)]
    pub(crate) samples: usize,

    /// A stable name recorded in the JSON artifact, such as `before-policy-cache`.
    #[arg(long)]
    pub(crate) name: Option<String>,

    /// Write the complete machine-readable JSON report to this path.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,

    /// Also print a concise human-readable summary.
    #[arg(long)]
    pub(crate) human: bool,
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
        BenchmarkCommand::Iteration(args) => {
            let report = iteration::full_training_iteration(&args)?;
            write_iteration(report, &args)
        }
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
    write_report(report, args.output.as_deref(), args.human)
}

fn write_iteration(report: crate::report::BenchmarkReport, args: &IterationArgs) -> Result<()> {
    write_report(report, args.output.as_deref(), args.human)
}

fn write_report(
    report: crate::report::BenchmarkReport,
    output: Option<&std::path::Path>,
    human: bool,
) -> Result<()> {
    let json = serde_json::to_string_pretty(&report)?;
    println!("{json}");

    if human {
        print_human(&report);
    }

    if let Some(path) = output {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create benchmark output directory {}",
                    parent.display()
                )
            })?;
        }
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
