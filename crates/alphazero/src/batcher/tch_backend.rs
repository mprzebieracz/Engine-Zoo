use super::{CombinedEncodedBatch, InferenceBackend};
use crate::evaluator::Evaluation;
use crate::network::ModelSpec;
use anyhow::{anyhow, Context, Result};
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
    states_buf: Option<Tensor>,
    index_buf: Option<Tensor>,
    gathered_buf: Option<Tensor>,
    value_buf: Option<Tensor>,
}

enum LoadedModel {
    Native {
        vs: nn::VarStore,
        net: crate::network::Network,
    },
    TensorRtTorchScript(CModule),
}

impl TchInferenceBackend {
    pub fn new(
        spec: ModelSpec,
        weights: &Path,
        device: Device,
        precision: InferencePrecision,
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
            model: LoadedModel::Native { vs, net },
            spec,
            device,
            input_kind: match precision {
                InferencePrecision::Fp32 => Kind::Float,
                InferencePrecision::Fp16 => Kind::Half,
            },
            precision,
            states_buf: None,
            index_buf: None,
            gathered_buf: None,
            value_buf: None,
        })
    }

    pub fn new_tensor_rt_torchscript(spec: ModelSpec, module: &Path, device: Device) -> Result<Self> {
        anyhow::ensure!(device.is_cuda(), "TensorRT inference requires CUDA");
        anyhow::ensure!(module.is_file(), "missing TensorRT TorchScript module: {}", module.display());
        let mut module = CModule::load_on_device(module, device)
            .with_context(|| format!("loading TensorRT TorchScript module from {}", module.display()))?;
        module.set_eval();
        Ok(Self {
            model: LoadedModel::TensorRtTorchScript(module),
            spec,
            device,
            // TensorRT chooses FP16 kernels internally. Its exported module
            // contract uses FP32 inputs and outputs.
            input_kind: Kind::Float,
            precision: InferencePrecision::Fp32,
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
}

impl InferenceBackend for TchInferenceBackend {
    fn reload_weights(&mut self, weights: &Path) -> Result<()> {
        match &mut self.model {
            LoadedModel::Native { vs, .. } => {
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
        let states_host = Self::staging(
            &mut self.states_buf,
            min_rows,
            c * h * w,
            Kind::Float,
            self.device,
        );
        unsafe {
            let dst = std::slice::from_raw_parts_mut(
                states_host.data_ptr() as *mut f32,
                (states_host.size()[0] * states_host.size()[1]) as usize,
            );
            dst[..batch.states.len()].copy_from_slice(&batch.states);
        }
        // Host staging remains FP32. The device copy happens before the FP16
        // cast, which preserves pinned-transfer performance and correct dtype.
        let inputs = states_host
            .narrow(0, 0, n)
            .view([n, c, h, w])
            .to_device_(self.device, Kind::Float, true, false)
            .to_kind(self.input_kind);
        let (policy, value) = match &self.model {
            LoadedModel::Native { net, .. } => {
                let output = net.forward_t(&inputs, false);
                (output.policy_logits, output.value.expected_value().to_kind(Kind::Float))
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
                let value = output.narrow(1, action_space, 1).squeeze_dim(1).to_kind(Kind::Float);
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
            let width = gathered_host.size()[1] as usize;
            unsafe {
                let indexes = std::slice::from_raw_parts_mut(
                    index_host.data_ptr() as *mut i64,
                    (index_host.size()[0] * index_host.size()[1]) as usize,
                );
                for row in 0..batch.len() {
                    let dst = &mut indexes[row * width..(row + 1) * width];
                    dst.fill(0);
                    let begin = batch.offsets[row] as usize;
                    let end = batch.offsets[row + 1] as usize;
                    for (slot, action) in dst.iter_mut().zip(&batch.legal_actions[begin..end]) {
                        *slot = i64::from(action.as_u32());
                    }
                }
            }
            let index_gpu =
                index_host
                    .narrow(0, 0, n)
                    .to_device_(self.device, Kind::Int64, true, false);
            let gathered_gpu = policy.gather(1, &index_gpu, false).to_kind(Kind::Float);
            gathered_host.narrow(0, 0, n).copy_(&gathered_gpu);
            Some((gathered_host, width))
        };
        let value_host = Self::staging(&mut self.value_buf, min_rows, 1, Kind::Float, self.device);
        value_host.narrow(0, 0, n).copy_(&value.view([n, 1]));
        let values =
            unsafe { std::slice::from_raw_parts(value_host.data_ptr() as *const f32, batch.len()) };
        (0..batch.len())
            .map(|row| -> Result<Evaluation> {
                let begin = batch.offsets[row] as usize;
                let end = batch.offsets[row + 1] as usize;
                let logits = gathered
                    .as_ref()
                    .map_or_else(Vec::new, |(host, width)| unsafe {
                        std::slice::from_raw_parts(
                            host.data_ptr() as *const f32,
                            host.size()[0] as usize * *width,
                        )[row * *width..row * *width + end - begin]
                            .to_vec()
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
