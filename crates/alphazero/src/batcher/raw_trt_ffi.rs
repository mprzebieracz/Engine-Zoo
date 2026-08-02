use anyhow::{bail, ensure, Result};
use half::f16;
use std::cell::Cell;
use std::ffi::{c_char, c_void, CStr};
use std::marker::PhantomData;
use std::ptr::NonNull;
use tch::{Device, Kind, Tensor};

const RAW_TRT_OK: i32 = 0;
const RAW_TRT_F32: i32 = 1;
const RAW_TRT_F16: i32 = 2;

#[repr(C)]
struct FfiSession {
    _private: [u8; 0],
}

#[repr(C)]
struct FfiExpectedContract {
    device_ordinal: i32,
    channels: i32,
    height: i32,
    width: i32,
    output_columns: i32,
    configured_max_batch: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FfiMetadata {
    input_dtype: i32,
    output_dtype: i32,
    channels: i32,
    height: i32,
    width: i32,
    output_columns: i32,
    min_batch: i32,
    opt_batch: i32,
    max_batch: i32,
}

unsafe extern "C" {
    fn raw_trt_session_create_from_blob(
        data: *const u8,
        size: usize,
        expected: *const FfiExpectedContract,
        out_session: *mut *mut FfiSession,
        out_metadata: *mut FfiMetadata,
    ) -> i32;

    fn raw_trt_session_run_host(
        session: *mut FfiSession,
        pinned_host_input: *const c_void,
        input_bytes: usize,
        batch_size: i32,
        device_output: *mut c_void,
        output_bytes: usize,
    ) -> i32;

    fn raw_trt_session_destroy(session: *mut FfiSession);
    fn raw_trt_last_error() -> *const c_char;
}

#[derive(Clone, Copy)]
pub(super) struct ExpectedContract {
    pub device: Device,
    pub state_shape: [i64; 3],
    pub output_columns: i64,
    pub configured_max_batch: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RawTrtMetadata {
    pub input_kind: Kind,
    pub output_kind: Kind,
    pub state_shape: [i64; 3],
    pub output_columns: i64,
    pub min_batch: usize,
    #[allow(dead_code)]
    pub opt_batch: usize,
    pub max_batch: usize,
}

pub(super) struct RawTrtSession {
    raw: NonNull<FfiSession>,
    metadata: RawTrtMetadata,
    device: Device,
    _not_sync: PhantomData<Cell<()>>,
}

// A session is exclusively mutated through `&mut self` and may move to the
// batcher executor thread. TensorRT execution contexts are not shared.
unsafe impl Send for RawTrtSession {}

impl RawTrtSession {
    pub(super) fn from_engine_bytes(bytes: &[u8], expected: ExpectedContract) -> Result<Self> {
        ensure!(!bytes.is_empty(), "raw TensorRT engine blob is empty");

        let device_ordinal = match expected.device {
            Device::Cuda(ordinal) => i32::try_from(ordinal)?,
            _ => bail!("raw TensorRT inference requires CUDA"),
        };
        let [channels, height, width] = expected.state_shape.map(i32::try_from);
        let ffi_expected = FfiExpectedContract {
            device_ordinal,
            channels: channels?,
            height: height?,
            width: width?,
            output_columns: i32::try_from(expected.output_columns)?,
            configured_max_batch: i32::try_from(expected.configured_max_batch)?,
        };

        let mut raw = std::ptr::null_mut();
        let mut metadata = FfiMetadata::default();
        let status = unsafe {
            raw_trt_session_create_from_blob(
                bytes.as_ptr(),
                bytes.len(),
                &ffi_expected,
                &mut raw,
                &mut metadata,
            )
        };
        check_status(status, "creating raw TensorRT session")?;

        let raw = NonNull::new(raw)
            .ok_or_else(|| anyhow::anyhow!("raw TensorRT returned a null successful session"))?;
        let metadata = match convert_metadata(metadata) {
            Ok(metadata) => metadata,
            Err(error) => {
                unsafe { raw_trt_session_destroy(raw.as_ptr()) };
                return Err(error);
            }
        };

        Ok(Self {
            raw,
            metadata,
            device: expected.device,
            _not_sync: PhantomData,
        })
    }

    pub(super) fn run_host(
        &mut self,
        pinned_input: &Tensor,
        batch_size: usize,
        device_output: &Tensor,
    ) -> Result<()> {
        self.validate_input(pinned_input, batch_size)?;
        self.validate_output(device_output, batch_size)?;

        let input_bytes = tensor_bytes(pinned_input);
        let output_bytes = tensor_bytes(device_output);
        let status = unsafe {
            raw_trt_session_run_host(
                self.raw.as_ptr(),
                pinned_input.data_ptr(),
                input_bytes,
                i32::try_from(batch_size)?,
                device_output.data_ptr(),
                output_bytes,
            )
        };

        check_status(status, "running raw TensorRT session")
    }

    pub(super) fn metadata(&self) -> RawTrtMetadata {
        self.metadata
    }

    fn validate_input(&self, input: &Tensor, batch_size: usize) -> Result<()> {
        ensure!(
            input.device() == Device::Cpu,
            "raw TensorRT input must be on CPU"
        );
        ensure!(
            input.is_pinned(self.device),
            "raw TensorRT input must be pinned"
        );
        ensure!(
            input.is_contiguous(),
            "raw TensorRT input must be contiguous"
        );
        ensure!(
            input.kind() == self.metadata.input_kind,
            "raw TensorRT input dtype mismatch"
        );

        let expected = [
            i64::try_from(batch_size)?,
            self.metadata.state_shape[0],
            self.metadata.state_shape[1],
            self.metadata.state_shape[2],
        ];
        ensure!(
            input.size() == expected,
            "raw TensorRT input shape mismatch"
        );

        ensure!(
            batch_size >= self.metadata.min_batch && batch_size <= self.metadata.max_batch,
            "raw TensorRT batch {batch_size} is outside profile [{}, {}]",
            self.metadata.min_batch,
            self.metadata.max_batch
        );

        Ok(())
    }

    fn validate_output(&self, output: &Tensor, batch_size: usize) -> Result<()> {
        ensure!(
            output.device() == self.device,
            "raw TensorRT output is on the wrong device"
        );
        ensure!(
            output.is_contiguous(),
            "raw TensorRT output must be contiguous"
        );
        ensure!(
            output.kind() == self.metadata.output_kind,
            "raw TensorRT output dtype mismatch"
        );

        let required = batch_size
            .checked_mul(usize::try_from(self.metadata.output_columns)?)
            .ok_or_else(|| anyhow::anyhow!("raw TensorRT output size overflow"))?;
        ensure!(
            output.numel() >= required,
            "raw TensorRT output buffer is too small"
        );

        Ok(())
    }
}

pub(super) fn copy_f32_to_tensor(tensor: &Tensor, source: &[f32]) -> Result<()> {
    validate_cpu_tensor(tensor, Kind::Float, source.len())?;

    let destination =
        unsafe { std::slice::from_raw_parts_mut(tensor.data_ptr() as *mut f32, source.len()) };
    destination.copy_from_slice(source);

    Ok(())
}

pub(super) fn copy_f16_to_tensor(tensor: &Tensor, source: &[f32]) -> Result<()> {
    validate_cpu_tensor(tensor, Kind::Half, source.len())?;

    let destination =
        unsafe { std::slice::from_raw_parts_mut(tensor.data_ptr() as *mut f16, source.len()) };
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination = f16::from_f32(*source);
    }

    Ok(())
}

pub(super) fn with_i64_data_mut<T>(
    tensor: &Tensor,
    operation: impl FnOnce(&mut [i64]) -> T,
) -> Result<T> {
    validate_cpu_tensor(tensor, Kind::Int64, tensor.numel())?;

    let values =
        unsafe { std::slice::from_raw_parts_mut(tensor.data_ptr() as *mut i64, tensor.numel()) };

    Ok(operation(values))
}

pub(super) fn with_f32_data<T>(tensor: &Tensor, operation: impl FnOnce(&[f32]) -> T) -> Result<T> {
    validate_cpu_tensor(tensor, Kind::Float, tensor.numel())?;

    let values =
        unsafe { std::slice::from_raw_parts(tensor.data_ptr() as *const f32, tensor.numel()) };

    Ok(operation(values))
}

fn validate_cpu_tensor(tensor: &Tensor, kind: Kind, required_elements: usize) -> Result<()> {
    ensure!(tensor.device() == Device::Cpu, "tensor must be on CPU");
    ensure!(tensor.is_contiguous(), "tensor must be contiguous");
    ensure!(tensor.kind() == kind, "tensor dtype mismatch");
    ensure!(tensor.numel() >= required_elements, "tensor is too small");

    Ok(())
}

impl Drop for RawTrtSession {
    fn drop(&mut self) {
        unsafe { raw_trt_session_destroy(self.raw.as_ptr()) }
    }
}

fn convert_metadata(metadata: FfiMetadata) -> Result<RawTrtMetadata> {
    let input_kind = convert_dtype(metadata.input_dtype, "input")?;
    let output_kind = convert_dtype(metadata.output_dtype, "output")?;
    let min_batch = usize::try_from(metadata.min_batch)?;
    let opt_batch = usize::try_from(metadata.opt_batch)?;
    let max_batch = usize::try_from(metadata.max_batch)?;

    ensure!(
        min_batch > 0 && min_batch <= opt_batch && opt_batch <= max_batch,
        "raw TensorRT returned invalid profile metadata"
    );
    ensure!(
        metadata.channels > 0
            && metadata.height > 0
            && metadata.width > 0
            && metadata.output_columns > 1,
        "raw TensorRT returned invalid shape metadata"
    );

    Ok(RawTrtMetadata {
        input_kind,
        output_kind,
        state_shape: [
            i64::from(metadata.channels),
            i64::from(metadata.height),
            i64::from(metadata.width),
        ],
        output_columns: i64::from(metadata.output_columns),
        min_batch,
        opt_batch,
        max_batch,
    })
}

fn convert_dtype(dtype: i32, tensor: &str) -> Result<Kind> {
    match dtype {
        RAW_TRT_F32 => Ok(Kind::Float),
        RAW_TRT_F16 => Ok(Kind::Half),
        other => bail!("raw TensorRT returned unsupported {tensor} dtype {other}"),
    }
}

fn tensor_bytes(tensor: &Tensor) -> usize {
    tensor.numel() * tensor.kind().elt_size_in_bytes()
}

fn check_status(status: i32, operation: &str) -> Result<()> {
    if status == RAW_TRT_OK {
        return Ok(());
    }

    bail!("{operation} failed with status {status}: {}", last_error())
}

fn last_error() -> String {
    unsafe {
        let message = raw_trt_last_error();
        if message.is_null() {
            return "unknown raw TensorRT error".to_owned();
        }

        CStr::from_ptr(message).to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn session_is_send() {
        assert_send::<RawTrtSession>();
    }

    #[test]
    fn converts_metadata() {
        let metadata = convert_metadata(FfiMetadata {
            input_dtype: RAW_TRT_F16,
            output_dtype: RAW_TRT_F32,
            channels: 8,
            height: 8,
            width: 8,
            output_columns: 4_673,
            min_batch: 1,
            opt_batch: 32,
            max_batch: 64,
        })
        .unwrap();

        assert_eq!(metadata.input_kind, Kind::Half);
        assert_eq!(metadata.output_kind, Kind::Float);
        assert_eq!(metadata.state_shape, [8, 8, 8]);
        assert_eq!(metadata.output_columns, 4_673);
        assert_eq!(
            (metadata.min_batch, metadata.opt_batch, metadata.max_batch),
            (1, 32, 64)
        );
    }

    #[test]
    fn rejects_invalid_metadata() {
        let error = convert_metadata(FfiMetadata {
            input_dtype: RAW_TRT_F32,
            output_dtype: RAW_TRT_F32,
            channels: 8,
            height: 8,
            width: 8,
            output_columns: 4_673,
            min_batch: 64,
            opt_batch: 32,
            max_batch: 1,
        })
        .unwrap_err();

        assert!(error.to_string().contains("profile metadata"));
    }
}
