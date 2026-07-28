use crate::report::{
    BenchmarkReport, BenchmarkSample, BenchmarkSummary, BuildMetadata, GitMetadata, HostMetadata,
};
use anyhow::Result;
use serde_json::json;
use std::process::Command;

pub fn report() -> Result<BenchmarkReport> {
    Ok(BenchmarkReport {
        schema_version: 1,
        benchmark: "environment".into(),
        started_at: timestamp(),
        git: git(),
        build: build(),
        host: host(),
        workload: json!({}),
        samples: vec![BenchmarkSample {
            elapsed_ns: 0,
            operations: 0,
            metrics: json!({}),
        }],
        summary: BenchmarkSummary {
            repetitions: 0,
            total_operations: 0,
            minimum_ns: 0,
            maximum_ns: 0,
            mean_ns: 0.0,
            median_ns: 0,
            stddev_ns: 0.0,
        },
        reports: Vec::new(),
    })
}

pub fn timestamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|time| time.as_secs().to_string())
        .unwrap_or_default()
}

pub fn git() -> GitMetadata {
    GitMetadata {
        commit: command("git", &["rev-parse", "HEAD"]),
        dirty: git_dirty(),
    }
}

pub fn build() -> BuildMetadata {
    BuildMetadata {
        cargo_profile: cargo_profile().into(),
        rust_version: command("rustc", &["--version"]),
        target: option_env!("TARGET").unwrap_or("unknown").into(),
        tch_version: "0.20".into(),
    }
}

fn cargo_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    }
    else {
        "release"
    }
}

pub fn host() -> HostMetadata {
    HostMetadata {
        os: std::env::consts::OS.into(),
        kernel: command("uname", &["-r"]),
        cpu_model: cpu_model(),
        cpu_count: std::thread::available_parallelism().map_or(1, usize::from),
        gpu: command("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]),
    }
}

fn git_dirty() -> Option<bool> {
    Command::new("git")
        .args(["diff", "--quiet"])
        .status()
        .ok()
        .map(|status| !status.success())
}

fn cpu_model() -> Option<String> {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("model name\t: ").map(str::to_owned))
}

fn command(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_detection_allows_no_cuda() {
        let detected = host();
        assert!(!detected.os.is_empty());
        assert!(detected.cpu_count > 0);
    }
}
