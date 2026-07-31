// Minimal C ABI over the TensorRT C++ runtime.
//
// TensorRT ships a C++-only API (no stable C API), so this shim exists purely
// to give the `raw_trt_backend` Rust module a narrow, stable, `extern "C"`
// surface: deserialize an engine, discover its I/O tensors, bind CUDA device
// pointers already owned by `tch::Tensor`, and run one synchronous inference
// pass. It intentionally does not manage any device memory: all buffers are
// allocated and owned by the Rust side (reusing the existing pinned-staging
// infrastructure), and only their raw device pointers cross this boundary.
#ifndef ENGINE_ZOO_RAW_TRT_RUNTIME_H
#define ENGINE_ZOO_RAW_TRT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RawTrtEngine RawTrtEngine;
typedef struct RawTrtContext RawTrtContext;

// Deserializes a serialized engine plan produced by
// `scripts/compile_tensorrt_raw.py`. Returns null on failure; call
// `raw_trt_last_error()` for a description.
RawTrtEngine *raw_trt_load_engine(const uint8_t *data, size_t size);
void raw_trt_free_engine(RawTrtEngine *engine);

int32_t raw_trt_num_io_tensors(const RawTrtEngine *engine);
// Returned pointer is owned by the engine and stays valid until it is freed.
const char *raw_trt_io_tensor_name(const RawTrtEngine *engine, int32_t index);
// 1 = input, 2 = output, 0 = neither (`nvinfer1::TensorIOMode`).
int32_t raw_trt_tensor_io_mode(const RawTrtEngine *engine, const char *name);
// `nvinfer1::DataType` ordinal; this backend only supports kFLOAT (0) and
// kHALF (1) tensors.
int32_t raw_trt_tensor_data_type(const RawTrtEngine *engine, const char *name);

RawTrtContext *raw_trt_create_context(RawTrtEngine *engine);
void raw_trt_free_context(RawTrtContext *context);
// Dedicated non-default CUDA stream owned by the context. Valid until
// `raw_trt_free_context`.
void *raw_trt_context_stream(RawTrtContext *context);

// `dims`/`nb_dims` describe the concrete shape used for this batch; it must
// be within the engine's compiled optimization profile.
int32_t raw_trt_set_input_shape(RawTrtContext *context, const char *name,
                                 const int64_t *dims, int32_t nb_dims);
// `device_ptr` must already reside on the CUDA device the engine was loaded
// for; the caller (Rust) retains ownership.
int32_t raw_trt_set_tensor_address(RawTrtContext *context, const char *name,
                                    void *device_ptr);
// Reads back a tensor's concrete shape for the input shapes set so far.
// `dims_out` must have room for at least 8 entries.
int32_t raw_trt_get_tensor_shape(RawTrtContext *context, const char *name,
                                  int64_t *dims_out, int32_t *nb_dims_out);

// Enqueues inference on `stream` (a `cudaStream_t`). Pass null to use the
// context's dedicated non-default stream. Returns 0 on success.
int32_t raw_trt_enqueue(RawTrtContext *context, void *stream);
// Blocks until all work queued on `stream` completes. Returns 0 on success.
int32_t raw_trt_synchronize_stream(void *stream);

// Thread-local description of the most recent failure from this file, valid
// until the next call into it on the same thread.
const char *raw_trt_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // ENGINE_ZOO_RAW_TRT_RUNTIME_H
