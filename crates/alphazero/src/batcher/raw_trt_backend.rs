//! Raw TensorRT runtime backend: a variant of [`super::TchInferenceBackend`]'s
//! TensorRT path that skips Torch-TensorRT/TorchScript entirely.
//!
//! The engine is built out-of-process by `scripts/compile_tensorrt_raw.py`
//! (ONNX export + the TensorRT Python builder API with a genuine persistent
//! timing cache) and loaded here through a small C shim
//! (`native/raw_trt_runtime.cpp`) over the TensorRT C++ runtime. Buffers are
//! still owned and allocated by `tch`/LibTorch (reusing the same pinned
//! host-staging pattern as [`super::TchInferenceBackend`]); only their raw
//! CUDA device pointers cross the FFI boundary. This keeps the CUDA context
//! and allocator shared with the rest of the process.
//!
//! This first version is deliberately synchronous (it fully synchronizes the
//! device around each `enqueueV3`), matching today's established
//! non-overlapped batcher execution model. Async multi-stream overlap is a
//! natural follow-up once this is validated for correctness and throughput
//! parity.

use super::{CombinedEncodedBatch, InferenceBackend};
use crate::evaluator::Evaluation;
use crate::network::ModelSpec;
use anyhow::{anyhow, bail, ensure, Context, Result};
use half::f16;
use std::collections::BTreeSet;
use std::ffi::{CStr, CString};
use std::path::Path;
use tch::{Device, Kind, Tensor};

mod ffi {
    use std::os::raw::{c_char, c_void};

    #[repr(C)]
    pub struct RawTrtEngine {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct RawTrtContext {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn raw_trt_load_engine(data: *const u8, size: usize) -> *mut RawTrtEngine;
        pub fn raw_trt_free_engine(engine: *mut RawTrtEngine);
        pub fn raw_trt_num_io_tensors(engine: *const RawTrtEngine) -> i32;
        pub fn raw_trt_io_tensor_name(engine: *const RawTrtEngine, index: i32) -> *const c_char;
        pub fn raw_trt_tensor_io_mode(engine: *const RawTrtEngine, name: *const c_char) -> i32;
        pub fn raw_trt_tensor_data_type(engine: *const RawTrtEngine, name: *const c_char) -> i32;
        pub fn raw_trt_create_context(engine: *mut RawTrtEngine) -> *mut RawTrtContext;
        pub fn raw_trt_free_context(context: *mut RawTrtContext);
        pub fn raw_trt_context_stream(context: *mut RawTrtContext) -> *mut c_void;
        pub fn raw_trt_set_input_shape(
            context: *mut RawTrtContext,
            name: *const c_char,
            dims: *const i64,
            nb_dims: i32,
        ) -> i32;
        pub fn raw_trt_set_tensor_address(
            context: *mut RawTrtContext,
            name: *const c_char,
            device_ptr: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn raw_trt_get_tensor_shape(
            context: *mut RawTrtContext,
            name: *const c_char,
            dims_out: *mut i64,
            nb_dims_out: *mut i32,
        ) -> i32;
        pub fn raw_trt_enqueue(context: *mut RawTrtContext, stream: *mut c_void) -> i32;
        pub fn raw_trt_synchronize_stream(stream: *mut c_void) -> i32;
        pub fn raw_trt_last_error() -> *const c_char;
    }
}

const TENSOR_IO_MODE_INPUT: i32 = 1;
const TENSOR_IO_MODE_OUTPUT: i32 = 2;
const TENSOR_DATA_TYPE_FLOAT: i32 = 0;
const TENSOR_DATA_TYPE_HALF: i32 = 1;

// Matches the packed contract already used by the Torch-TensorRT TorchScript
// path (`TchInferenceBackend::LoadedModel::TensorRtTorchScript` and
// `tensor_rt::export_torchscript`): one input, one output holding policy
// logits followed by a single scalar-value column. Reusing this contract
// means both backends can be built from the exact same TorchScript export
// artifact and are directly numerically comparable.
const INPUT_TENSOR_NAME: &str = "input";
const OUTPUT_TENSOR_NAME: &str = "output";

fn last_error() -> String {
    unsafe {
        let ptr = ffi::raw_trt_last_error();
        if ptr.is_null() {
            return "unknown raw TensorRT error".to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

struct EngineHandle(*mut ffi::RawTrtEngine);
// TensorRT's engine object is safe to share across threads once built; this
// backend only ever accesses it from the single batcher executor thread.
unsafe impl Send for EngineHandle {}
impl Drop for EngineHandle {
    fn drop(&mut self) {
        unsafe { ffi::raw_trt_free_engine(self.0) }
    }
}

struct ContextHandle(*mut ffi::RawTrtContext);
unsafe impl Send for ContextHandle {}
impl Drop for ContextHandle {
    fn drop(&mut self) {
        unsafe { ffi::raw_trt_free_context(self.0) }
    }
}

/// Loads and runs a raw TensorRT engine plan (not a TorchScript module).
pub struct RawTensorRtBackend {
    context: ContextHandle,
    // Declared after `context` so Drop destroys the execution context before
    // the engine (TensorRT requires that order).
    _engine: EngineHandle,
    spec: ModelSpec,
    device: Device,
    io_kind: Kind,
    input_name: CString,
    output_name: CString,
    states_buf: Option<Tensor>,
    // Device-resident buffer bound to TensorRT's packed output address.
    output_buf: Option<Tensor>,
    // Pinned host buffer the CPU-visible value column is copied back into.
    value_host_buf: Option<Tensor>,
    index_buf: Option<Tensor>,
    gathered_buf: Option<Tensor>,
}

impl RawTensorRtBackend {
    /// Loads a serialized engine plan produced by
    /// `scripts/compile_tensorrt_raw.py`. The plan must have been built for
    /// this exact TensorRT version and GPU.
    pub fn new(spec: ModelSpec, engine_path: &Path, device: Device) -> Result<Self> {
        anyhow::ensure!(device.is_cuda(), "raw TensorRT inference requires CUDA");
        ensure!(
            engine_path.is_file(),
            "missing TensorRT engine plan: {}",
            engine_path.display()
        );

        // Forces the CUDA primary context for `device` to exist and be
        // current on this thread before any TensorRT runtime call. Engine
        // construction may happen on a different thread than `evaluate()`;
        // both need the same device selected.
        let _touch = Tensor::zeros([1], (Kind::Float, device));

        let bytes = std::fs::read(engine_path)
            .with_context(|| format!("reading TensorRT engine plan {}", engine_path.display()))?;

        let engine_ptr = unsafe { ffi::raw_trt_load_engine(bytes.as_ptr(), bytes.len()) };
        if engine_ptr.is_null() {
            bail!("failed to deserialize TensorRT engine: {}", last_error());
        }
        let engine = EngineHandle(engine_ptr);

        let io_kind = Self::validate_io_contract(&engine)?;

        let context_ptr = unsafe { ffi::raw_trt_create_context(engine.0) };
        if context_ptr.is_null() {
            bail!(
                "failed to create TensorRT execution context: {}",
                last_error()
            );
        }
        let context = ContextHandle(context_ptr);

        Ok(Self {
            context,
            _engine: engine,
            spec,
            device,
            io_kind,
            input_name: CString::new(INPUT_TENSOR_NAME).expect("static name has no NUL bytes"),
            output_name: CString::new(OUTPUT_TENSOR_NAME).expect("static name has no NUL bytes"),
            states_buf: None,
            output_buf: None,
            value_host_buf: None,
            index_buf: None,
            gathered_buf: None,
        })
    }

    /// Confirms the engine has exactly the two named tensors this backend
    /// expects (FP32 or FP16). Returns the LibTorch kind used for bindings.
    fn validate_io_contract(engine: &EngineHandle) -> Result<Kind> {
        let count = unsafe { ffi::raw_trt_num_io_tensors(engine.0) };
        ensure!(count >= 0, "failed to enumerate TensorRT engine tensors");

        let mut seen = BTreeSet::new();
        let mut io_kind: Option<Kind> = None;
        for index in 0..count {
            let name_ptr = unsafe { ffi::raw_trt_io_tensor_name(engine.0, index) };
            ensure!(!name_ptr.is_null(), "TensorRT engine tensor name was null");
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();
            let data_type = unsafe { ffi::raw_trt_tensor_data_type(engine.0, name_ptr) };
            let kind = match data_type {
                TENSOR_DATA_TYPE_FLOAT => Kind::Float,
                TENSOR_DATA_TYPE_HALF => Kind::Half,
                other => bail!(
                    "TensorRT tensor '{name}' must be FP32 or FP16, found data type {other}"
                ),
            };
            match io_kind {
                None => io_kind = Some(kind),
                Some(existing) if existing == kind => {}
                Some(_) => bail!("TensorRT input/output tensors must share one dtype"),
            }
            let io_mode = unsafe { ffi::raw_trt_tensor_io_mode(engine.0, name_ptr) };
            let expected_mode = match name.as_str() {
                INPUT_TENSOR_NAME => TENSOR_IO_MODE_INPUT,
                OUTPUT_TENSOR_NAME => TENSOR_IO_MODE_OUTPUT,
                other => bail!(
                    "unexpected TensorRT tensor '{other}'; engines built for this backend must \
                     only have '{INPUT_TENSOR_NAME}' and '{OUTPUT_TENSOR_NAME}'"
                ),
            };
            ensure!(
                io_mode == expected_mode,
                "TensorRT tensor '{name}' has unexpected IO mode {io_mode}"
            );
            seen.insert(name);
        }
        ensure!(
            seen.len() == 2
                && seen.contains(INPUT_TENSOR_NAME)
                && seen.contains(OUTPUT_TENSOR_NAME),
            "TensorRT engine is missing one of '{INPUT_TENSOR_NAME}', '{OUTPUT_TENSOR_NAME}'"
        );
        Ok(io_kind.expect("engine has I/O tensors"))
    }

    /// Mirrors `TchInferenceBackend::staging`: grow-only host buffer, pinned
    /// for DMA transfer.
    fn staging(slot: &mut Option<Tensor>, rows: i64, cols: i64, kind: Kind, device: Device) -> Tensor {
        let needs_alloc = match slot {
            Some(t) => t.size()[0] < rows || t.size()[1] < cols || t.kind() != kind,
            None => true,
        };
        if needs_alloc {
            let old_cols = slot.as_ref().map_or(0, |t| t.size()[1]);
            let tensor = Tensor::zeros([rows, cols.max(old_cols)], (kind, Device::Cpu));
            *slot = Some(tensor.pin_memory(device));
        }
        slot.as_ref().unwrap().shallow_clone()
    }

    /// Grow-only device-resident buffer for TensorRT output bindings.
    fn device_buffer(
        slot: &mut Option<Tensor>,
        rows: i64,
        cols: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let needs_alloc = match slot {
            Some(t) => t.size()[0] < rows || t.size()[1] < cols || t.kind() != kind,
            None => true,
        };
        if needs_alloc {
            let old_cols = slot.as_ref().map_or(0, |t| t.size()[1]);
            *slot = Some(Tensor::zeros([rows, cols.max(old_cols)], (kind, device)));
        }
        slot.as_ref().unwrap().shallow_clone()
    }

    fn active(buffer: &Tensor, rows: i64, columns: i64) -> Tensor {
        buffer.narrow(0, 0, rows).narrow(1, 0, columns)
    }

    fn copy_states_as_fp32(states_host: &Tensor, states: &[f32]) {
        let dst =
            unsafe { std::slice::from_raw_parts_mut(states_host.data_ptr() as *mut f32, states.len()) };
        dst.copy_from_slice(states);
    }

    fn copy_states_as_fp16(states_host: &Tensor, states: &[f32]) {
        let dst =
            unsafe { std::slice::from_raw_parts_mut(states_host.data_ptr() as *mut f16, states.len()) };
        for (dst, src) in dst.iter_mut().zip(states) {
            *dst = f16::from_f32(*src);
        }
    }

    fn gathered_logits(gathered: &Tensor, row: usize, legal_actions: usize) -> Vec<f32> {
        let row_stride = gathered.stride()[0] as usize;
        let rows = gathered.size()[0] as usize;
        let active_columns = gathered.size()[1] as usize;
        let len = (rows - 1) * row_stride + active_columns;
        let values = unsafe { std::slice::from_raw_parts(gathered.data_ptr() as *const f32, len) };
        values[row * row_stride..row * row_stride + legal_actions].to_vec()
    }

    /// Runs the network on an already-staged, contiguous
    /// `[n, channels, height, width]` FP32 host tensor and returns
    /// `(policy, value)` device tensors with shapes `[n, action_size]` and
    /// `[n]`. This skips the legal-action gather `evaluate()` performs, so
    /// it measures pure network latency comparable to the native and
    /// Torch-TensorRT `forward()` helpers in `engine-bench`'s
    /// `raw-inference` benchmark. Production self-play uses `evaluate()`.
    pub fn forward(&mut self, host_states: &Tensor) -> Result<(Tensor, Tensor)> {
        tch::no_grad(|| {
            let inputs = host_states
                .to_device_(self.device, Kind::Float, true, false)
                .to_kind(self.io_kind);
            let (policy, value) = self.run_engine(&inputs)?;
            Ok((
                policy.to_kind(Kind::Float),
                value.to_kind(Kind::Float).squeeze_dim(1),
            ))
        })
    }

    /// Binds `inputs` (already CUDA-resident, shape `[n, c, h, w]`, matching
    /// `io_kind`) and runs the engine, returning `(policy, value)` views into
    /// the shared output buffer with shapes `[n, action_size]` and `[n, 1]`.
    fn run_engine(&mut self, inputs: &Tensor) -> Result<(Tensor, Tensor)> {
        let size = inputs.size();
        anyhow::ensure!(size.len() == 4, "raw TensorRT input must be rank 4");
        let [n, c, h, w] = [size[0], size[1], size[2], size[3]];
        let action_size = self.spec.action_size() as i64;
        let output_columns = action_size + 1;

        let output =
            Self::device_buffer(&mut self.output_buf, n, output_columns, self.io_kind, self.device);
        let output_active = Self::active(&output, n, output_columns);

        self.bind_and_run(inputs, [n, c, h, w], &output_active)?;

        let policy = output_active.narrow(1, 0, action_size);
        let value = output_active.narrow(1, action_size, 1);
        Ok((policy, value))
    }

    fn evaluate_inner(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>> {
        let rows = batch.len();
        let n = rows as i64;
        let [c, h, w] = self.spec.state_shape();

        let states_host =
            Self::staging(&mut self.states_buf, n, c * h * w, self.io_kind, self.device);
        match self.io_kind {
            Kind::Half => Self::copy_states_as_fp16(&states_host, &batch.states),
            _ => Self::copy_states_as_fp32(&states_host, &batch.states),
        }
        let inputs = states_host
            .narrow(0, 0, n)
            .view([n, c, h, w])
            .to_device_(self.device, self.io_kind, true, false);

        let (policy, value) = self.run_engine(&inputs)?;
        let policy = policy.to_kind(Kind::Float);
        let value = value.to_kind(Kind::Float);

        let max_actions = batch
            .offsets
            .windows(2)
            .map(|range| (range[1] - range[0]) as i64)
            .max()
            .unwrap_or(0);
        self.gather_evaluations(batch, policy, value, n, max_actions)
    }

    fn bind_and_run(&mut self, input: &Tensor, input_dims: [i64; 4], output: &Tensor) -> Result<()> {
        let dims = input_dims;
        let status = unsafe {
            ffi::raw_trt_set_input_shape(
                self.context.0,
                self.input_name.as_ptr(),
                dims.as_ptr(),
                dims.len() as i32,
            )
        };
        ensure!(
            status == 0,
            "setting TensorRT input shape failed: {}",
            last_error()
        );

        for (name, tensor) in [(&self.input_name, input), (&self.output_name, output)] {
            let status = unsafe {
                ffi::raw_trt_set_tensor_address(
                    self.context.0,
                    name.as_ptr(),
                    tensor.data_ptr(),
                )
            };
            ensure!(
                status == 0,
                "binding TensorRT tensor address failed: {}",
                last_error()
            );
        }

        let stream = unsafe { ffi::raw_trt_context_stream(self.context.0) };
        ensure!(
            !stream.is_null(),
            "TensorRT context has no CUDA stream: {}",
            last_error()
        );

        let status = unsafe { ffi::raw_trt_enqueue(self.context.0, stream) };
        ensure!(status == 0, "TensorRT enqueue failed: {}", last_error());

        let status = unsafe { ffi::raw_trt_synchronize_stream(stream) };
        ensure!(status == 0, "TensorRT stream sync failed: {}", last_error());

        Ok(())
    }

    fn gather_evaluations(
        &mut self,
        batch: &CombinedEncodedBatch,
        policy: Tensor,
        value: Tensor,
        n: i64,
        max_actions: i64,
    ) -> Result<Vec<Evaluation>> {
        let gathered = if max_actions == 0 {
            None
        } else {
            let index_host = Self::staging(&mut self.index_buf, n, max_actions, Kind::Int64, self.device);
            let gathered_host_slot = &mut self.gathered_buf;
            let index_active = Self::active(&index_host, n, max_actions);
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
            let gathered_gpu = policy.gather(1, &index_gpu, false);
            let gathered_host = Self::staging(gathered_host_slot, n, max_actions, Kind::Float, self.device);
            let mut gathered_active = Self::active(&gathered_host, n, max_actions);
            gathered_active.copy_(&gathered_gpu);
            Some(gathered_active)
        };

        let value_host = Self::staging(&mut self.value_host_buf, n, 1, Kind::Float, self.device);
        value_host.narrow(0, 0, n).copy_(&value.view([n, 1]));
        let values =
            unsafe { std::slice::from_raw_parts(value_host.data_ptr() as *const f32, batch.len()) };

        (0..batch.len())
            .map(|row| -> Result<Evaluation> {
                let begin = batch.offsets[row] as usize;
                let end = batch.offsets[row + 1] as usize;
                let logits = gathered
                    .as_ref()
                    .map_or_else(Vec::new, |host| Self::gathered_logits(host, row, end - begin));
                Ok(Evaluation {
                    logits,
                    value: search::PositionValue::new(values[row])
                        .map_err(|_| anyhow!("network returned invalid value at row {row}"))?,
                })
            })
            .collect::<Result<Vec<_>>>()
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
        anyhow::ensure!(
            batch.states.len()
                == batch.len() * self.spec.state_shape().iter().product::<i64>() as usize,
            "encoded state shape does not match model specification"
        );
        tch::no_grad(|| self.evaluate_inner(batch))
    }
}
