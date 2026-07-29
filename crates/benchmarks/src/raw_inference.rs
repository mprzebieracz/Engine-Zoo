//! Raw, synchronous CUDA inference measurements for a fixed chess checkpoint.
//!
//! This intentionally lives in `engine-bench`: it measures alternate host
//! staging paths without adding benchmark policy to production inference.

use crate::cli::RawInferenceArgs;
use crate::device;
use crate::harness;
use crate::report::{BenchmarkReport, BenchmarkSample};
use alphazero::{ChessHistory, ExperimentConfig, ModelSpec, Network};
use anyhow::{Context, Result};
use serde_json::json;
use std::path::Path;
use std::time::Instant;
use tch::{nn, CModule, Device, IValue, Kind, Tensor};

#[derive(Clone, Copy)]
enum NativeStaging {
    Fp32,
    Fp16,
}

impl NativeStaging {
    const fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32-host-staging",
            Self::Fp16 => "fp16-host-staging",
        }
    }
}

/// Runs native FP16 with both staging variants and a TensorRT TorchScript
/// module for every requested batch size. Timed samples include host-to-device
/// transfer and forward execution, and synchronize CUDA at each boundary.
pub fn chess_h4(args: &RawInferenceArgs) -> Result<BenchmarkReport> {
    anyhow::ensure!(
        !args.batch_sizes.is_empty(),
        "--batch-sizes must not be empty"
    );
    anyhow::ensure!(args.samples > 0, "--samples must be positive");
    anyhow::ensure!(
        args.checkpoint.is_file(),
        "missing checkpoint: {}",
        args.checkpoint.display()
    );
    anyhow::ensure!(
        args.tensor_rt_module.is_file(),
        "missing TensorRT module: {}",
        args.tensor_rt_module.display()
    );

    let device = device::select(crate::device::DeviceKind::Cuda)?;
    let experiment = ExperimentConfig::read_toml(&args.experiment)
        .with_context(|| format!("reading experiment {}", args.experiment.display()))?;
    require_h4(&experiment.model)?;

    let native = NativeFp16::load(&experiment.model, &args.checkpoint, device)?;
    let tensor_rt = TensorRt::load(&args.tensor_rt_module, device)?;
    let [channels, height, width] = experiment.model.state_shape();
    let mut reports = Vec::with_capacity(args.batch_sizes.len() * 3);

    for &batch_size in &args.batch_sizes {
        anyhow::ensure!(batch_size > 0, "batch sizes must be positive");
        let host_states = host_states(batch_size, channels, height, width, device);
        let host_states_fp16 = host_states.to_kind(Kind::Half).pin_memory(device);

        reports.push(native_report(
            &native,
            &host_states,
            NativeStaging::Fp32,
            args.warmup,
            args.samples,
            batch_size,
        )?);
        reports.push(native_report(
            &native,
            &host_states_fp16,
            NativeStaging::Fp16,
            args.warmup,
            args.samples,
            batch_size,
        )?);
        reports.push(tensor_rt_report(
            &tensor_rt,
            &native,
            &host_states,
            args.warmup,
            args.samples,
            batch_size,
        )?);
    }

    harness::combine("raw-inference.chess-h4", reports)
}

fn require_h4(spec: &ModelSpec) -> Result<()> {
    match spec {
        ModelSpec::ChessSe {
            history: ChessHistory::Four,
            ..
        } => Ok(()),
        _ => anyhow::bail!("raw-inference is intentionally scoped to the chess SE H4 model"),
    }
}

fn host_states(rows: usize, channels: i64, height: i64, width: i64, device: Device) -> Tensor {
    let host = Tensor::randn(
        [rows as i64, channels, height, width],
        (Kind::Float, Device::Cpu),
    );
    host.pin_memory(device)
}

struct NativeFp16 {
    _store: nn::VarStore,
    network: Network,
    device: Device,
}

impl NativeFp16 {
    fn load(spec: &ModelSpec, checkpoint: &Path, device: Device) -> Result<Self> {
        let mut store = nn::VarStore::new(device);
        let network = Network::new(&store.root(), spec)?;
        store
            .load(checkpoint)
            .with_context(|| format!("loading checkpoint {}", checkpoint.display()))?;
        store.half();

        Ok(Self {
            _store: store,
            network,
            device,
        })
    }

    fn forward(&self, host_states: &Tensor, staging: NativeStaging) -> (Tensor, Tensor) {
        let input = match staging {
            NativeStaging::Fp32 => host_states
                .to_device_(self.device, Kind::Float, true, false)
                .to_kind(Kind::Half),
            NativeStaging::Fp16 => host_states.to_device_(self.device, Kind::Half, true, false),
        };
        let output = self.network.forward_t(&input, false);
        (
            output.policy_logits,
            output.value.expected_value().to_kind(Kind::Float),
        )
    }
}

struct TensorRt {
    module: CModule,
    device: Device,
}

impl TensorRt {
    fn load(path: &Path, device: Device) -> Result<Self> {
        let mut module = CModule::load_on_device(path, device)
            .with_context(|| format!("loading TensorRT module {}", path.display()))?;
        module.set_eval();
        Ok(Self { module, device })
    }

    fn forward(&self, host_states: &Tensor) -> Result<(Tensor, Tensor)> {
        let input = host_states.to_device_(self.device, Kind::Float, true, false);
        let packed: Tensor = self
            .module
            .forward_is(&[IValue::from(input)])?
            .try_into()
            .context("TensorRT TorchScript forward must return a packed tensor")?;
        let actions = packed.size()[1] - 1;
        Ok((
            packed.narrow(1, 0, actions),
            packed.narrow(1, actions, 1).squeeze_dim(1),
        ))
    }
}

fn native_report(
    native: &NativeFp16,
    host_states: &Tensor,
    staging: NativeStaging,
    warmup: usize,
    samples: usize,
    batch_size: usize,
) -> Result<BenchmarkReport> {
    measure_forward(
        &format!("raw-inference.chess-h4.native-fp16.{}", staging.name()),
        warmup,
        samples,
        batch_size,
        json!({
            "backend": "native-tch",
            "precision": "fp16",
            "host_staging": staging.name(),
            "cuda_memory": "unsupported-by-tch-public-api",
        }),
        || {
            device::synchronize(native.device);
            let started = Instant::now();
            let outputs = tch::no_grad(|| native.forward(host_states, staging));
            device::synchronize(native.device);
            let _ = std::hint::black_box(outputs);
            started.elapsed().as_nanos()
        },
    )
}

fn tensor_rt_report(
    tensor_rt: &TensorRt,
    native: &NativeFp16,
    host_states: &Tensor,
    warmup: usize,
    samples: usize,
    batch_size: usize,
) -> Result<BenchmarkReport> {
    let differences = numerical_difference(tensor_rt, native, host_states)?;
    measure_forward(
        "raw-inference.chess-h4.tensor-rt-fp16",
        warmup,
        samples,
        batch_size,
        json!({
            "backend": "tensor-rt-torch-script",
            "precision": "fp16-internal-fp32-contract",
            "host_staging": "fp32-host-staging",
            "numerical_difference_vs_native_fp16": differences,
            "cuda_memory": "unsupported-by-tch-public-api",
        }),
        || {
            device::synchronize(tensor_rt.device);
            let started = Instant::now();
            let outputs = tch::no_grad(|| tensor_rt.forward(host_states))
                .expect("validated TensorRT module remains executable");
            device::synchronize(tensor_rt.device);
            let _ = std::hint::black_box(outputs);
            started.elapsed().as_nanos()
        },
    )
}

fn measure_forward(
    name: &str,
    warmup: usize,
    samples: usize,
    batch_size: usize,
    metadata: serde_json::Value,
    mut forward: impl FnMut() -> u128,
) -> Result<BenchmarkReport> {
    harness::measure(
        name,
        &warmup.to_string(),
        samples,
        json!({ "batch_size": batch_size, "metadata": metadata }),
        || {
            let elapsed_ns = forward();
            BenchmarkSample {
                elapsed_ns,
                operations: batch_size as u64,
                metrics: json!({
                    "backend_latency_ms": elapsed_ns as f64 / 1_000_000.0,
                    "positions_per_second": batch_size as f64 * 1_000_000_000.0 / elapsed_ns.max(1) as f64,
                }),
            }
        },
    )
}

fn numerical_difference(
    tensor_rt: &TensorRt,
    native: &NativeFp16,
    host_states: &Tensor,
) -> Result<serde_json::Value> {
    let ((native_policy, native_value), trt_outputs) = tch::no_grad(|| {
        (
            native.forward(host_states, NativeStaging::Fp32),
            tensor_rt.forward(host_states),
        )
    });
    let (trt_policy, trt_value) = trt_outputs?;
    device::synchronize(native.device);

    Ok(json!({
        "policy_max_abs": max_abs_difference(&native_policy, &trt_policy),
        "policy_mean_abs": mean_abs_difference(&native_policy, &trt_policy),
        "value_max_abs": max_abs_difference(&native_value, &trt_value),
        "value_mean_abs": mean_abs_difference(&native_value, &trt_value),
    }))
}

fn max_abs_difference(reference: &Tensor, candidate: &Tensor) -> f64 {
    reference.subtract(candidate).abs().max().double_value(&[])
}

fn mean_abs_difference(reference: &Tensor, candidate: &Tensor) -> f64 {
    reference
        .subtract(candidate)
        .abs()
        .mean(Kind::Float)
        .double_value(&[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_benchmark_is_scoped_to_h4() {
        assert!(require_h4(&ModelSpec::chess_se(
            ChessHistory::Four,
            alphazero::ValueHeadSpec::Wdl { hidden: 128 }
        ))
        .is_ok());
        assert!(require_h4(&ModelSpec::chess_se(
            ChessHistory::One,
            alphazero::ValueHeadSpec::Wdl { hidden: 128 }
        ))
        .is_err());
    }
}
