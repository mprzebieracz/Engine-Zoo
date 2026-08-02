#ifndef ENGINE_ZOO_RAW_TRT_RUNTIME_H
#define ENGINE_ZOO_RAW_TRT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define RAW_TRT_NOEXCEPT noexcept
#else
#define RAW_TRT_NOEXCEPT
#endif

typedef struct RawTrtSession RawTrtSession;

typedef enum RawTrtStatus {
  RAW_TRT_OK = 0,
  RAW_TRT_INVALID_ARGUMENT = 1,
  RAW_TRT_IO_ERROR = 2,
  RAW_TRT_DESERIALIZE_ERROR = 3,
  RAW_TRT_CONTRACT_ERROR = 4,
  RAW_TRT_CUDA_ERROR = 5,
  RAW_TRT_EXECUTION_ERROR = 6,
  RAW_TRT_INTERNAL_ERROR = 7
} RawTrtStatus;

typedef enum RawTrtDType {
  RAW_TRT_F32 = 1,
  RAW_TRT_F16 = 2
} RawTrtDType;

typedef struct RawTrtExpectedContract {
  int32_t device_ordinal;
  int32_t channels;
  int32_t height;
  int32_t width;
  int32_t output_columns;
  int32_t configured_max_batch;
} RawTrtExpectedContract;

typedef struct RawTrtMetadata {
  RawTrtDType input_dtype;
  RawTrtDType output_dtype;
  int32_t channels;
  int32_t height;
  int32_t width;
  int32_t output_columns;
  int32_t min_batch;
  int32_t opt_batch;
  int32_t max_batch;
} RawTrtMetadata;

RawTrtStatus raw_trt_session_create_from_blob(
    const uint8_t *data, size_t size, const RawTrtExpectedContract *expected,
    RawTrtSession **out_session, RawTrtMetadata *out_metadata) RAW_TRT_NOEXCEPT;

RawTrtStatus raw_trt_session_run_host(
    RawTrtSession *session, const void *pinned_host_input, size_t input_bytes,
    int32_t batch_size, void *device_output,
    size_t output_bytes) RAW_TRT_NOEXCEPT;

void raw_trt_session_destroy(RawTrtSession *session) RAW_TRT_NOEXCEPT;

const char *raw_trt_last_error(void) RAW_TRT_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#undef RAW_TRT_NOEXCEPT

#endif
