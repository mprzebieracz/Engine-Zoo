use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Deserialize, Serialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub benchmark: String,
    pub started_at: String,
    pub git: GitMetadata,
    pub build: BuildMetadata,
    pub host: HostMetadata,
    pub workload: Value,
    pub samples: Vec<BenchmarkSample>,
    pub summary: BenchmarkSummary,
    #[serde(default)]
    pub reports: Vec<BenchmarkReport>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct GitMetadata {
    pub commit: Option<String>,
    pub dirty: Option<bool>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct BuildMetadata {
    pub cargo_profile: String,
    pub rust_version: Option<String>,
    pub target: String,
    pub tch_version: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct HostMetadata {
    pub os: String,
    pub kernel: Option<String>,
    pub cpu_model: Option<String>,
    pub cpu_count: usize,
    pub gpu: Option<String>,
    #[serde(default)]
    pub gpu_driver: Option<String>,
    #[serde(default)]
    pub cuda_available: bool,
    #[serde(default)]
    pub cuda_device_count: i64,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct BenchmarkSample {
    pub elapsed_ns: u128,
    pub operations: u64,
    pub metrics: Value,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct BenchmarkSummary {
    pub repetitions: usize,
    pub total_operations: u64,
    pub minimum_ns: u128,
    pub maximum_ns: u128,
    pub mean_ns: f64,
    pub median_ns: u128,
    pub stddev_ns: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn report_json_round_trips() {
        let report = BenchmarkReport {
            schema_version: 1,
            benchmark: "test".into(),
            started_at: "0".into(),
            git: GitMetadata {
                commit: Some("abc".into()),
                dirty: Some(false),
            },
            build: BuildMetadata {
                cargo_profile: "release".into(),
                rust_version: Some("rustc".into()),
                target: "target".into(),
                tch_version: env!("ENGINE_BENCH_TCH_VERSION").into(),
            },
            host: HostMetadata {
                os: "linux".into(),
                kernel: None,
                cpu_model: None,
                cpu_count: 1,
                gpu: None,
                gpu_driver: None,
                cuda_available: false,
                cuda_device_count: 0,
            },
            workload: json!({ "fixed": true }),
            samples: vec![BenchmarkSample {
                elapsed_ns: 1,
                operations: 2,
                metrics: json!({}),
            }],
            summary: BenchmarkSummary {
                repetitions: 1,
                total_operations: 2,
                minimum_ns: 1,
                maximum_ns: 1,
                mean_ns: 1.0,
                median_ns: 1,
                stddev_ns: 0.0,
            },
            reports: Vec::new(),
        };

        let encoded = serde_json::to_string(&report).unwrap();
        let decoded: BenchmarkReport = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.benchmark, report.benchmark);
        assert_eq!(decoded.samples[0].operations, report.samples[0].operations);
        assert!(decoded.reports.is_empty());
    }
}
