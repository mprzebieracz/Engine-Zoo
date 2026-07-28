use anyhow::{bail, Result};
use clap::ValueEnum;
use tch::{Cuda, Device};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum DeviceKind {
    /// Run on the first CUDA device. This is the benchmark default.
    #[default]
    Cuda,
    /// Run on the host CPU.
    Cpu,
    /// Prefer CUDA, falling back to CPU when it is unavailable.
    Auto,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum Precision {
    #[default]
    Fp32,
    Fp16,
}

impl Precision {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
        }
    }
}

pub fn select(kind: DeviceKind) -> Result<Device> {
    match kind {
        DeviceKind::Cuda => {
            if !Cuda::is_available() {
                bail!("CUDA was requested for this benchmark, but tch::Cuda::is_available() is false. This benchmark does not fall back silently; use --device cpu or --device auto. Check that the loaded LibTorch is CUDA-enabled and compatible with the installed NVIDIA driver (nvidia-smi alone is not sufficient).")
            }
            Ok(Device::Cuda(0))
        }
        DeviceKind::Cpu => Ok(Device::Cpu),
        DeviceKind::Auto => Ok(Device::cuda_if_available()),
    }
}

pub fn validate_precision(precision: Precision, device: Device) -> Result<()> {
    if precision == Precision::Fp16 && !device.is_cuda() {
        bail!("--precision fp16 requires a CUDA benchmark device")
    }
    Ok(())
}

pub fn name(device: Device) -> String {
    match device {
        Device::Cpu => "cpu".into(),
        Device::Cuda(index) => format!("cuda:{index}"),
        Device::Mps => "mps".into(),
        Device::Vulkan => "vulkan".into(),
    }
}

/// Wait only at a measurement boundary, so asynchronous CUDA launches are
/// included in the elapsed time without leaking synchronization into engines.
pub fn synchronize(device: Device) {
    if let Device::Cuda(index) = device {
        Cuda::synchronize(index as i64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_selection_never_needs_cuda() {
        assert_eq!(select(DeviceKind::Cpu).unwrap(), Device::Cpu);
    }

    #[test]
    fn auto_selection_always_resolves() {
        assert!(matches!(
            select(DeviceKind::Auto).unwrap(),
            Device::Cpu | Device::Cuda(_)
        ));
    }
}
