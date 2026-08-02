//! Raw TensorRT inference using a project-owned native session.

use super::{CombinedEncodedBatch, InferenceBackend};
use crate::evaluator::Evaluation;
use crate::network::ModelSpec;
use anyhow::{anyhow, bail, ensure, Context, Result};
use std::path::Path;
use tch::{Device, Kind, Tensor};

#[path = "raw_trt_ffi.rs"]
mod raw_trt_ffi;

use raw_trt_ffi::{
    copy_f16_to_tensor, copy_f32_to_tensor, with_f32_data, with_i64_data_mut, ExpectedContract,
    RawTrtSession,
};

/// Loads and runs a raw TensorRT engine plan (not a TorchScript module).
pub struct RawTensorRtBackend {
    session: RawTrtSession,
    spec: ModelSpec,
    device: Device,
    input_kind: Kind,
    output_kind: Kind,
    states_buf: Option<Tensor>,
    output_buf: Option<Tensor>,
    value_host_buf: Option<Tensor>,
    index_buf: Option<Tensor>,
    gathered_buf: Option<Tensor>,
}

impl RawTensorRtBackend {
    /// Loads an engine and accepts any batch within its compiled profile.
    pub fn new(spec: ModelSpec, engine_path: &Path, device: Device) -> Result<Self> {
        Self::new_with_max_batch(spec, engine_path, device, 0)
    }

    /// Loads an engine and verifies the batcher's configured maximum against
    /// its TensorRT optimization profile.
    pub fn new_with_max_batch(
        spec: ModelSpec,
        engine_path: &Path,
        device: Device,
        configured_max_batch: usize,
    ) -> Result<Self> {
        ensure!(device.is_cuda(), "raw TensorRT inference requires CUDA");
        ensure!(
            engine_path.is_file(),
            "missing TensorRT engine plan: {}",
            engine_path.display()
        );

        // Initialize LibTorch's primary context before the TensorRT runtime
        // selects the same device.
        let _touch = Tensor::zeros([1], (Kind::Float, device));
        let bytes = std::fs::read(engine_path)
            .with_context(|| format!("reading TensorRT engine plan {}", engine_path.display()))?;

        let state_shape = spec.state_shape();
        let output_columns = i64::try_from(spec.action_size())? + 1;
        let session = RawTrtSession::from_engine_bytes(
            &bytes,
            ExpectedContract {
                device,
                state_shape,
                output_columns,
                configured_max_batch,
            },
        )?;
        let metadata = session.metadata();

        ensure!(
            metadata.state_shape == state_shape,
            "raw TensorRT state shape mismatch"
        );
        ensure!(
            metadata.output_columns == output_columns,
            "raw TensorRT output width mismatch"
        );

        Ok(Self {
            session,
            spec,
            device,
            input_kind: metadata.input_kind,
            output_kind: metadata.output_kind,
            states_buf: None,
            output_buf: None,
            value_host_buf: None,
            index_buf: None,
            gathered_buf: None,
        })
    }

    fn staging(
        slot: &mut Option<Tensor>,
        rows: i64,
        columns: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let needs_allocation = match slot {
            Some(tensor) => {
                tensor.size()[0] < rows || tensor.size()[1] < columns || tensor.kind() != kind
            }
            None => true,
        };

        if needs_allocation {
            let old_columns = slot.as_ref().map_or(0, |tensor| tensor.size()[1]);
            let tensor = Tensor::zeros([rows, columns.max(old_columns)], (kind, Device::Cpu));
            *slot = Some(tensor.pin_memory(device));
        }

        slot.as_ref().unwrap().shallow_clone()
    }

    fn device_buffer(
        slot: &mut Option<Tensor>,
        rows: i64,
        columns: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let needs_allocation = match slot {
            Some(tensor) => {
                tensor.size()[0] < rows
                    || tensor.size()[1] < columns
                    || tensor.kind() != kind
                    || tensor.device() != device
            }
            None => true,
        };

        if needs_allocation {
            let old_columns = slot.as_ref().map_or(0, |tensor| tensor.size()[1]);
            *slot = Some(Tensor::zeros(
                [rows, columns.max(old_columns)],
                (kind, device),
            ));
        }

        slot.as_ref().unwrap().shallow_clone()
    }

    fn active(buffer: &Tensor, rows: i64, columns: i64) -> Tensor {
        buffer.narrow(0, 0, rows).narrow(1, 0, columns)
    }

    fn gathered_logits(
        gathered: &Tensor,
        values: &[f32],
        row: usize,
        legal_actions: usize,
    ) -> Vec<f32> {
        let row_stride = gathered.stride()[0] as usize;

        values[row * row_stride..row * row_stride + legal_actions].to_vec()
    }

    /// Runs a CPU state tensor without an intermediate CUDA input tensor.
    pub fn forward(&mut self, host_states: &Tensor) -> Result<(Tensor, Tensor)> {
        tch::no_grad(|| {
            ensure!(
                host_states.device() == Device::Cpu,
                "raw TensorRT input must be on CPU"
            );
            ensure!(
                host_states.is_contiguous(),
                "raw TensorRT input must be contiguous"
            );

            let size = host_states.size();
            ensure!(size.len() == 4, "raw TensorRT input must be rank 4");
            ensure!(
                size[1..] == self.spec.state_shape(),
                "raw TensorRT input shape mismatch"
            );

            let rows = size[0];
            let columns = self.spec.state_shape().iter().product();
            let pinned = Self::staging(
                &mut self.states_buf,
                rows,
                columns,
                self.input_kind,
                self.device,
            );
            let mut active = Self::active(&pinned, rows, columns);
            active.copy_(&host_states.view([rows, columns]).to_kind(self.input_kind));
            let input = active.view(size.as_slice());

            let (policy, value) = self.run_engine(&input)?;

            Ok((
                policy.to_kind(Kind::Float),
                value.to_kind(Kind::Float).squeeze_dim(1),
            ))
        })
    }

    fn run_engine(&mut self, pinned_input: &Tensor) -> Result<(Tensor, Tensor)> {
        let rows = pinned_input.size()[0];
        let action_size = i64::try_from(self.spec.action_size())?;
        let output_columns = action_size + 1;
        let output = Self::device_buffer(
            &mut self.output_buf,
            rows,
            output_columns,
            self.output_kind,
            self.device,
        );
        let output = Self::active(&output, rows, output_columns);

        self.session
            .run_host(pinned_input, usize::try_from(rows)?, &output)?;

        let policy = output.narrow(1, 0, action_size);
        let value = output.narrow(1, action_size, 1);

        Ok((policy, value))
    }

    fn evaluate_inner(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>> {
        let rows = i64::try_from(batch.len())?;
        let [channels, height, width] = self.spec.state_shape();
        let columns = channels * height * width;
        let states_host = Self::staging(
            &mut self.states_buf,
            rows,
            columns,
            self.input_kind,
            self.device,
        );
        let states_host = Self::active(&states_host, rows, columns);

        match self.input_kind {
            Kind::Half => copy_f16_to_tensor(&states_host, &batch.states)?,
            Kind::Float => copy_f32_to_tensor(&states_host, &batch.states)?,
            _ => unreachable!("session creation rejects other input dtypes"),
        }

        let input = states_host.view([rows, channels, height, width]);
        let (policy, value) = self.run_engine(&input)?;
        let value = value.to_kind(Kind::Float);
        let max_actions = batch
            .offsets
            .windows(2)
            .map(|range| i64::from(range[1] - range[0]))
            .max()
            .unwrap_or(0);

        self.gather_evaluations(batch, policy, value, rows, max_actions)
    }

    fn gather_evaluations(
        &mut self,
        batch: &CombinedEncodedBatch,
        policy: Tensor,
        value: Tensor,
        rows: i64,
        max_actions: i64,
    ) -> Result<Vec<Evaluation>> {
        let gathered = self.gather_legal_logits(batch, &policy, rows, max_actions)?;
        let value_host = Self::staging(&mut self.value_host_buf, rows, 1, Kind::Float, self.device);
        value_host.narrow(0, 0, rows).copy_(&value.view([rows, 1]));

        with_f32_data(&value_host, |values| match gathered.as_ref() {
            Some(gathered) => with_f32_data(gathered, |logits| {
                Self::build_evaluations(batch, values, Some((gathered, logits)))
            })?,
            None => Self::build_evaluations(batch, values, None),
        })?
    }

    fn build_evaluations(
        batch: &CombinedEncodedBatch,
        values: &[f32],
        gathered: Option<(&Tensor, &[f32])>,
    ) -> Result<Vec<Evaluation>> {
        (0..batch.len())
            .map(|row| {
                let begin = batch.offsets[row] as usize;
                let end = batch.offsets[row + 1] as usize;
                let logits = gathered.map_or_else(Vec::new, |(tensor, values)| {
                    Self::gathered_logits(tensor, values, row, end - begin)
                });
                let value = search::PositionValue::new(values[row])
                    .map_err(|_| anyhow!("network returned invalid value at row {row}"))?;

                Ok(Evaluation { logits, value })
            })
            .collect()
    }

    fn gather_legal_logits(
        &mut self,
        batch: &CombinedEncodedBatch,
        policy: &Tensor,
        rows: i64,
        max_actions: i64,
    ) -> Result<Option<Tensor>> {
        if max_actions == 0 {
            return Ok(None);
        }

        let index_host = Self::staging(
            &mut self.index_buf,
            rows,
            max_actions,
            Kind::Int64,
            self.device,
        );
        let index_active = Self::active(&index_host, rows, max_actions);
        self.fill_legal_indexes(batch, &index_active)?;

        let index_gpu = index_active.to_device_(self.device, Kind::Int64, true, false);
        // Convert only the gathered legal logits, never the full policy.
        let gathered_gpu = policy.gather(1, &index_gpu, false).to_kind(Kind::Float);
        let gathered_host = Self::staging(
            &mut self.gathered_buf,
            rows,
            max_actions,
            Kind::Float,
            self.device,
        );
        let mut gathered_active = Self::active(&gathered_host, rows, max_actions);
        gathered_active.copy_(&gathered_gpu);

        Ok(Some(gathered_active))
    }

    fn fill_legal_indexes(&self, batch: &CombinedEncodedBatch, indexes: &Tensor) -> Result<()> {
        let row_stride = indexes.stride()[0] as usize;
        let active_width = indexes.size()[1] as usize;

        with_i64_data_mut(indexes, |storage| {
            for row in 0..batch.len() {
                let destination = &mut storage[row * row_stride..row * row_stride + active_width];
                destination.fill(0);

                let begin = batch.offsets[row] as usize;
                let end = batch.offsets[row + 1] as usize;
                for (slot, action) in destination.iter_mut().zip(&batch.legal_actions[begin..end]) {
                    *slot = i64::from(action.as_u32());
                }
            }
        })
    }
}

impl InferenceBackend for RawTensorRtBackend {
    fn reload_weights(&mut self, _weights: &Path) -> Result<()> {
        bail!(
            "raw TensorRT engines have fixed weights; compile a new engine for the next \
             checkpoint and start a new run"
        )
    }

    fn evaluate(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        ensure!(
            batch.states.len()
                == batch.len() * self.spec.state_shape().iter().product::<i64>() as usize,
            "encoded state shape does not match model specification"
        );

        tch::no_grad(|| self.evaluate_inner(batch))
    }
}
