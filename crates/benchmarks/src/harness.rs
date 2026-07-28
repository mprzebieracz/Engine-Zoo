use crate::environment;
use crate::report::{BenchmarkReport, BenchmarkSample, BenchmarkSummary};
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

pub fn measure<F>(
    benchmark: &str,
    warmup: &str,
    samples: usize,
    workload: Value,
    mut work: F,
) -> Result<BenchmarkReport>
where
    F: FnMut() -> BenchmarkSample,
{
    if samples == 0 {
        bail!("--samples must be positive");
    }

    run_warmup(warmup, &mut work)?;

    let samples: Vec<_> = (0..samples).map(|_| work()).collect();
    let summary = summarize(&samples);

    Ok(BenchmarkReport {
        schema_version: 1,
        benchmark: benchmark.into(),
        started_at: environment::timestamp(),
        git: environment::git(),
        build: environment::build(),
        host: environment::host(),
        workload,
        samples,
        summary,
        reports: Vec::new(),
    })
}

pub fn combine(benchmark: &str, reports: Vec<BenchmarkReport>) -> Result<BenchmarkReport> {
    Ok(BenchmarkReport {
        schema_version: 1,
        benchmark: benchmark.into(),
        started_at: environment::timestamp(),
        git: environment::git(),
        build: environment::build(),
        host: environment::host(),
        workload: json!({ "benchmarks": reports.iter().map(|report| &report.benchmark).collect::<Vec<_>>() }),
        samples: Vec::new(),
        summary: BenchmarkSummary {
            repetitions: 0,
            total_operations: 0,
            minimum_ns: 0,
            maximum_ns: 0,
            mean_ns: 0.0,
            median_ns: 0,
            stddev_ns: 0.0,
        },
        reports,
    })
}

fn run_warmup<F>(value: &str, work: &mut F) -> Result<()>
where
    F: FnMut() -> BenchmarkSample,
{
    if let Some(duration) = parse_duration(value) {
        let started = Instant::now();
        while started.elapsed() < duration {
            std::hint::black_box(work());
        }
        return Ok(());
    }

    let iterations: usize = value.parse().map_err(|_| {
        anyhow::anyhow!("--warmup must be an iteration count or duration ending in ms/s")
    })?;
    for _ in 0..iterations {
        std::hint::black_box(work());
    }
    Ok(())
}

fn parse_duration(value: &str) -> Option<Duration> {
    value
        .strip_suffix("ms")
        .and_then(|milliseconds| milliseconds.parse().ok())
        .map(Duration::from_millis)
        .or_else(|| {
            value
                .strip_suffix('s')
                .and_then(|seconds| seconds.parse().ok())
                .map(Duration::from_secs)
        })
}

fn summarize(samples: &[BenchmarkSample]) -> BenchmarkSummary {
    let mut elapsed = samples
        .iter()
        .map(|sample| sample.elapsed_ns)
        .collect::<Vec<_>>();
    elapsed.sort_unstable();

    let repetitions = elapsed.len();
    let total_operations = samples.iter().map(|sample| sample.operations).sum();
    let mean_ns = elapsed.iter().sum::<u128>() as f64 / repetitions.max(1) as f64;
    let variance = elapsed
        .iter()
        .map(|value| (*value as f64 - mean_ns).powi(2))
        .sum::<f64>()
        / repetitions.max(1) as f64;

    BenchmarkSummary {
        repetitions,
        total_operations,
        minimum_ns: elapsed.first().copied().unwrap_or(0),
        maximum_ns: elapsed.last().copied().unwrap_or(0),
        mean_ns,
        median_ns: elapsed.get(repetitions / 2).copied().unwrap_or(0),
        stddev_ns: variance.sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_exclude_warmup_samples() {
        let report = measure("test", "3", 2, json!({}), || BenchmarkSample {
            elapsed_ns: 1,
            operations: 1,
            metrics: json!({}),
        })
        .unwrap();
        assert_eq!(report.samples.len(), 2);
        assert_eq!(report.summary.total_operations, 2);
    }

    #[test]
    fn suite_keeps_workload_samples_separate() {
        let child = measure("child", "0", 1, json!({}), || BenchmarkSample {
            elapsed_ns: 1,
            operations: 1,
            metrics: json!({}),
        })
        .unwrap();
        let suite = combine("suite", vec![child]).unwrap();

        assert!(suite.samples.is_empty());
        assert_eq!(suite.reports.len(), 1);
        assert_eq!(suite.reports[0].summary.total_operations, 1);
    }
}
