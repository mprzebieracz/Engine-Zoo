use super::{CombinedEncodedBatch, InferenceBackend};
use crate::evaluator::Evaluation;
use crate::network::ModelSpec;
use anyhow::{anyhow, Context, Result};
use half::f16;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tch::{nn, CModule, Device, IValue, Kind, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferencePrecision {
    Fp32,
    Fp16,
}

/// LibTorch implementation. Dynamic batching deliberately does not leak into
/// this type: it only receives one contiguous batch at a time.
pub struct TchInferenceBackend {
    model: LoadedModel,
    spec: ModelSpec,
    device: Device,
    input_kind: Kind,
    precision: InferencePrecision,
    fp16_host_staging: bool,
    states_buf: Option<Tensor>,
    index_buf: Option<Tensor>,
    gathered_buf: Option<Tensor>,
    value_buf: Option<Tensor>,
}

enum LoadedModel {
    Native(Box<NativeModel>),
    TensorRtTorchScript(CModule),
}

struct NativeModel {
    vs: nn::VarStore,
    net: crate::network::Network,
}

impl TchInferenceBackend {
    pub fn new(
        spec: ModelSpec,
        weights: &Path,
        device: Device,
        precision: InferencePrecision,
    ) -> Result<Self> {
        Self::new_with_fp16_host_staging(spec, weights, device, precision, false)
    }

    /// Enables the optional A/B path that converts state planes to FP16 on the
    /// CPU before copying the pinned buffer to CUDA.
    pub fn new_with_fp16_host_staging(
        spec: ModelSpec,
        weights: &Path,
        device: Device,
        precision: InferencePrecision,
        fp16_host_staging: bool,
    ) -> Result<Self> {
        anyhow::ensure!(
            precision == InferencePrecision::Fp32 || device.is_cuda(),
            "FP16 inference is only supported on CUDA"
        );
        let mut vs = nn::VarStore::new(device);
        let net = crate::network::Network::new(&vs.root(), &spec)?;
        vs.load(weights)
            .with_context(|| format!("loading network weights from {}", weights.display()))?;
        if precision == InferencePrecision::Fp16 {
            vs.half();
        }
        Ok(Self {
            model: LoadedModel::Native(Box::new(NativeModel { vs, net })),
            spec,
            device,
            input_kind: match precision {
                InferencePrecision::Fp32 => Kind::Float,
                InferencePrecision::Fp16 => Kind::Half,
            },
            precision,
            fp16_host_staging,
            states_buf: None,
            index_buf: None,
            gathered_buf: None,
            value_buf: None,
        })
    }

    pub fn new_tensor_rt_torchscript(
        spec: ModelSpec,
        module: &Path,
        device: Device,
    ) -> Result<Self> {
        anyhow::ensure!(device.is_cuda(), "TensorRT inference requires CUDA");
        anyhow::ensure!(
            module.is_file(),
            "missing TensorRT TorchScript module: {}",
            module.display()
        );
        let mut module = CModule::load_on_device(module, device).with_context(|| {
            format!(
                "loading TensorRT TorchScript module from {}",
                module.display()
            )
        })?;
        module.set_eval();
        Ok(Self {
            model: LoadedModel::TensorRtTorchScript(module),
            spec,
            device,
            // TensorRT chooses FP16 kernels internally. Its exported module
            // contract uses FP32 inputs and outputs.
            input_kind: Kind::Float,
            precision: InferencePrecision::Fp32,
            fp16_host_staging: false,
            states_buf: None,
            index_buf: None,
            gathered_buf: None,
            value_buf: None,
        })
    }

    /// Allocates grow-only host buffers, pinned on CUDA for DMA transfers.
    fn staging(
        slot: &mut Option<Tensor>,
        rows: i64,
        cols: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let needs_alloc = match slot {
            Some(t) => t.size()[0] < rows || t.size()[1] < cols,
            None => true,
        };
        if needs_alloc {
            let old_cols = slot.as_ref().map_or(0, |t| t.size()[1]);
            let tensor = Tensor::zeros([rows, cols.max(old_cols)], (kind, Device::Cpu));
            *slot = Some(if device.is_cuda() {
                tensor.pin_memory(device)
            }
            else {
                tensor
            });
        }
        slot.as_ref().unwrap().shallow_clone()
    }

    /// Narrows a grow-only staging allocation to the shape used by this batch.
    ///
    /// The view can retain a wider row stride than its visible width. Callers
    /// that access host memory directly must therefore use `stride()[0]` when
    /// moving between rows.
    fn active_staging(buffer: &Tensor, rows: i64, columns: i64) -> Tensor {
        buffer.narrow(0, 0, rows).narrow(1, 0, columns)
    }

    fn gathered_logits(gathered: &Tensor, row: usize, legal_actions: usize) -> Vec<f32> {
        debug_assert!(legal_actions <= gathered.size()[1] as usize);

        let row_stride = gathered.stride()[0] as usize;
        let rows = gathered.size()[0] as usize;
        let active_columns = gathered.size()[1] as usize;
        let len = (rows - 1) * row_stride + active_columns;
        let values = unsafe { std::slice::from_raw_parts(gathered.data_ptr() as *const f32, len) };

        values[row * row_stride..row * row_stride + legal_actions].to_vec()
    }

    fn use_fp16_host_staging(&self) -> bool {
        self.device.is_cuda()
            && self.precision == InferencePrecision::Fp16
            && self.fp16_host_staging
    }

    fn copy_states_as_fp16(states_host: &Tensor, states: &[f32]) {
        let dst = unsafe {
            std::slice::from_raw_parts_mut(states_host.data_ptr() as *mut f16, states.len())
        };

        for (dst, src) in dst.iter_mut().zip(states) {
            *dst = f16::from_f32(*src);
        }
    }

    fn copy_states_as_fp32(states_host: &Tensor, states: &[f32]) {
        let dst = unsafe {
            std::slice::from_raw_parts_mut(states_host.data_ptr() as *mut f32, states.len())
        };

        dst.copy_from_slice(states);
    }
}

impl InferenceBackend for TchInferenceBackend {
    fn reload_weights(&mut self, weights: &Path) -> Result<()> {
        match &mut self.model {
            LoadedModel::Native(native) => {
                let vs = &mut native.vs;
                vs.float();
                vs.load(weights)
                    .with_context(|| format!("loading network weights from {}", weights.display()))?;
                if self.precision == InferencePrecision::Fp16 {
                    vs.half();
                }
                Ok(())
            }
            LoadedModel::TensorRtTorchScript(_) => anyhow::bail!(
                "TensorRT TorchScript modules cannot reload safetensors weights; compile a module for the next checkpoint and start a new run"
            ),
        }
    }

    fn evaluate(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        anyhow::ensure!(
            batch.states.len()
                == batch.len() * self.spec.state_shape().iter().product::<i64>() as usize,
            "encoded state shape does not match model specification"
        );
        tch::no_grad(|| self.evaluate_inner(batch))
    }
}

impl TchInferenceBackend {
    fn evaluate_inner(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>> {
        let rows = batch.len();
        let n = rows as i64;
        let [c, h, w] = self.spec.state_shape();
        let min_rows = rows as i64;
        let state_kind = if self.use_fp16_host_staging() {
            Kind::Half
        }
        else {
            Kind::Float
        };
        let states_host = Self::staging(
            &mut self.states_buf,
            min_rows,
            c * h * w,
            state_kind,
            self.device,
        );
        let inputs = if self.use_fp16_host_staging() {
            Self::copy_states_as_fp16(&states_host, &batch.states);

            states_host.narrow(0, 0, n).view([n, c, h, w]).to_device_(
                self.device,
                Kind::Half,
                true,
                false,
            )
        }
        else {
            Self::copy_states_as_fp32(&states_host, &batch.states);

            // The default path preserves the established FP32 host upload and
            // GPU-side FP16 conversion for an apples-to-apples benchmark.
            states_host
                .narrow(0, 0, n)
                .view([n, c, h, w])
                .to_device_(self.device, Kind::Float, true, false)
                .to_kind(self.input_kind)
        };
        let (policy, value) = match &mut self.model {
            LoadedModel::Native(native) => {
                let output = native.net.forward_t(&inputs, false);
                (
                    output.policy_logits,
                    output.value.expected_value().to_kind(Kind::Float),
                )
            }
            LoadedModel::TensorRtTorchScript(module) => {
                let output = module
                    .forward_is(&[IValue::from(inputs)])
                    .context("executing TensorRT TorchScript module")?;
                let output: Tensor = output
                    .try_into()
                    .context("TensorRT TorchScript forward must return a packed tensor")?;
                let action_space = self.spec.action_size() as i64;
                anyhow::ensure!(
                    output.dim() == 2 && output.size()[1] == action_space + 1,
                    "TensorRT TorchScript output must have shape [batch, {}]",
                    action_space + 1
                );
                let policy = output.narrow(1, 0, action_space);
                let value = output
                    .narrow(1, action_space, 1)
                    .squeeze_dim(1)
                    .to_kind(Kind::Float);
                (policy, value)
            }
        };
        let max_actions = batch
            .offsets
            .windows(2)
            .map(|range| (range[1] - range[0]) as i64)
            .max()
            .unwrap_or(0);
        if self.device.is_cuda() {
            self.evaluate_cuda(batch, policy, value, n, min_rows, max_actions)
        }
        else {
            self.evaluate_cpu(batch, policy, value)
        }
    }

    fn evaluate_cuda(
        &mut self,
        batch: &CombinedEncodedBatch,
        policy: Tensor,
        value: Tensor,
        n: i64,
        min_rows: i64,
        max_actions: i64,
    ) -> Result<Vec<Evaluation>> {
        let gathered = if max_actions == 0 {
            None
        }
        else {
            let index_host = Self::staging(
                &mut self.index_buf,
                min_rows,
                max_actions,
                Kind::Int64,
                self.device,
            );
            let gathered_host = Self::staging(
                &mut self.gathered_buf,
                min_rows,
                max_actions,
                Kind::Float,
                self.device,
            );
            let index_active = Self::active_staging(&index_host, n, max_actions);
            let row_stride = index_active.stride()[0] as usize;
            let active_width = max_actions as usize;
            unsafe {
                let indexes = std::slice::from_raw_parts_mut(
                    index_host.data_ptr() as *mut i64,
                    (index_host.size()[0] * index_host.size()[1]) as usize,
                );
                for row in 0..batch.len() {
                    let dst = &mut indexes[row * row_stride..row * row_stride + active_width];
                    dst.fill(0);
                    let begin = batch.offsets[row] as usize;
                    let end = batch.offsets[row + 1] as usize;
                    for (slot, action) in dst.iter_mut().zip(&batch.legal_actions[begin..end]) {
                        *slot = i64::from(action.as_u32());
                    }
                }
            }
            let index_gpu = index_active.to_device_(self.device, Kind::Int64, true, false);
            let gathered_gpu = policy.gather(1, &index_gpu, false).to_kind(Kind::Float);
            let mut gathered_active = Self::active_staging(&gathered_host, n, max_actions);
            gathered_active.copy_(&gathered_gpu);
            Some(gathered_active)
        };
        let value_host = Self::staging(&mut self.value_buf, min_rows, 1, Kind::Float, self.device);
        value_host.narrow(0, 0, n).copy_(&value.view([n, 1]));
        let values =
            unsafe { std::slice::from_raw_parts(value_host.data_ptr() as *const f32, batch.len()) };
        (0..batch.len())
            .map(|row| -> Result<Evaluation> {
                let begin = batch.offsets[row] as usize;
                let end = batch.offsets[row + 1] as usize;
                let logits = gathered.as_ref().map_or_else(Vec::new, |host| {
                    Self::gathered_logits(host, row, end - begin)
                });
                Ok(Evaluation {
                    logits,
                    value: search::PositionValue::new(values[row])
                        .map_err(|_| anyhow!("network returned invalid value at row {row}"))?,
                })
            })
            .collect::<Result<Vec<_>>>()
    }

    fn evaluate_cpu(
        &self,
        batch: &CombinedEncodedBatch,
        policy: Tensor,
        value: Tensor,
    ) -> Result<Vec<Evaluation>> {
        let policy: Vec<f32> = policy
            .to_kind(Kind::Float)
            .contiguous()
            .view(-1)
            .try_into()?;
        let values: Vec<f32> = value.contiguous().view(-1).try_into()?;
        let action_size = self.spec.action_size();
        (0..batch.len())
            .map(|row| -> Result<Evaluation> {
                let begin = batch.offsets[row] as usize;
                let end = batch.offsets[row + 1] as usize;
                let logits = batch.legal_actions[begin..end]
                    .iter()
                    .map(|action| policy[row * action_size + action.index()])
                    .collect();
                Ok(Evaluation {
                    logits,
                    value: search::PositionValue::new(values[row])
                        .map_err(|_| anyhow!("network returned invalid value at row {row}"))?,
                })
            })
            .collect::<Result<Vec<_>>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_gathered_view_ignores_retained_staging_columns() {
        let mut staging = None;
        let mut wide = TchInferenceBackend::staging(&mut staging, 2, 4, Kind::Float, Device::Cpu);
        wide.copy_(
            &Tensor::from_slice(&[1.0f32, 2.0, 100.0, 101.0, 3.0, 4.0, 200.0, 201.0]).view([2, 4]),
        );

        let narrow = TchInferenceBackend::staging(&mut staging, 2, 2, Kind::Float, Device::Cpu);
        let mut active = TchInferenceBackend::active_staging(&narrow, 2, 2);
        active.copy_(&Tensor::from_slice(&[10.0f32, 11.0, 20.0, 21.0]).view([2, 2]));

        assert_eq!(narrow.size(), vec![2, 4]);
        assert_eq!(active.size(), vec![2, 2]);
        assert_eq!(
            TchInferenceBackend::gathered_logits(&active, 0, 2),
            [10.0, 11.0]
        );
        assert_eq!(
            TchInferenceBackend::gathered_logits(&active, 1, 2),
            [20.0, 21.0]
        );
        assert_eq!(TchInferenceBackend::gathered_logits(&active, 1, 1), [20.0]);
    }

    #[test]
    fn fp16_host_staging_converts_encoded_states_on_cpu() {
        let staging = Tensor::zeros([1, 4], (Kind::Half, Device::Cpu));
        let states = [0.0, 0.5, -1.0, 12.25];

        TchInferenceBackend::copy_states_as_fp16(&staging, &states);

        let actual: Vec<f32> = staging.to_kind(Kind::Float).view(-1).try_into().unwrap();
        assert_eq!(actual, states);
    }
}
