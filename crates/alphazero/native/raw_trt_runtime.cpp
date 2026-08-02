#include "raw_trt_runtime.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>

namespace {

thread_local std::string last_error_message;

void set_error(std::string message) noexcept {
  try {
    last_error_message = std::move(message);
  } catch (...) {
    last_error_message.clear();
  }
}

void set_error(const char *message) noexcept {
  try {
    last_error_message = message;
  } catch (...) {
    last_error_message.clear();
  }
}

void set_exception_error(const char *operation, const char *detail) noexcept {
  try {
    set_error(std::string(operation) + ": " + detail);
  } catch (...) {
    set_error("C++ exception crossed the raw TensorRT implementation boundary");
  }
}

class StderrLogger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[raw-tensor-rt] %s\n", message);
    }
  }
};

StderrLogger &logger() noexcept {
  static StderrLogger instance;
  return instance;
}

RawTrtStatus cuda_failure(const char *operation, cudaError_t error) noexcept {
  try {
    std::ostringstream message;
    message << operation << " failed (CUDA " << static_cast<int>(error)
            << "): " << cudaGetErrorString(error);
    set_error(message.str());
  } catch (...) {
    set_error("CUDA operation failed and its detailed error could not be formatted");
  }

  return RAW_TRT_CUDA_ERROR;
}

size_t dtype_size(RawTrtDType dtype) noexcept {
  return dtype == RAW_TRT_F16 ? 2 : 4;
}

bool checked_product(size_t left, size_t right, size_t &result) noexcept {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
    return false;
  }

  result = left * right;
  return true;
}

RawTrtStatus tensor_bytes(int32_t batch, int32_t columns, RawTrtDType dtype,
                          size_t &bytes) noexcept {
  size_t elements = 0;
  if (batch <= 0 || columns <= 0 ||
      !checked_product(static_cast<size_t>(batch),
                       static_cast<size_t>(columns), elements) ||
      !checked_product(elements, dtype_size(dtype), bytes)) {
    set_error("tensor byte size overflow");
    return RAW_TRT_INVALID_ARGUMENT;
  }

  return RAW_TRT_OK;
}

RawTrtStatus input_tensor_bytes(int32_t batch, int32_t channels,
                                int32_t height, int32_t width,
                                RawTrtDType dtype, size_t &bytes) noexcept {
  size_t elements = static_cast<size_t>(batch);
  for (int32_t dimension : {channels, height, width}) {
    if (batch <= 0 || dimension <= 0 ||
        !checked_product(elements, static_cast<size_t>(dimension), elements)) {
      set_error("input tensor byte size overflow");
      return RAW_TRT_INVALID_ARGUMENT;
    }
  }

  if (!checked_product(elements, dtype_size(dtype), bytes)) {
    set_error("input tensor byte size overflow");
    return RAW_TRT_INVALID_ARGUMENT;
  }

  return RAW_TRT_OK;
}

RawTrtStatus to_dtype(nvinfer1::DataType dtype, const char *tensor_name,
                      RawTrtDType &result) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    result = RAW_TRT_F32;
    return RAW_TRT_OK;
  case nvinfer1::DataType::kHALF:
    result = RAW_TRT_F16;
    return RAW_TRT_OK;
  default:
    set_error(std::string("TensorRT tensor '") + tensor_name +
              "' must use FP32 or FP16");
    return RAW_TRT_CONTRACT_ERROR;
  }
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() noexcept {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  RawTrtStatus ensure_capacity(size_t bytes) noexcept {
    if (bytes <= capacity_) {
      return RAW_TRT_OK;
    }

    size_t grown = capacity_;
    if (capacity_ <= std::numeric_limits<size_t>::max() / 2) {
      grown *= 2;
    }
    size_t capacity = std::max(bytes, grown);
    void *replacement = nullptr;
    const cudaError_t allocation = cudaMalloc(&replacement, capacity);
    if (allocation != cudaSuccess) {
      return cuda_failure("cudaMalloc(raw TensorRT input)", allocation);
    }

    if (pointer_ != nullptr) {
      const cudaError_t release = cudaFree(pointer_);
      if (release != cudaSuccess) {
        cudaFree(replacement);
        return cuda_failure("cudaFree(raw TensorRT input)", release);
      }
    }

    pointer_ = replacement;
    capacity_ = capacity;
    return RAW_TRT_OK;
  }

  void *data() noexcept { return pointer_; }

private:
  void *pointer_ = nullptr;
  size_t capacity_ = 0;
};

struct DiscoveredContract {
  std::string input_name;
  std::string output_name;
  RawTrtMetadata metadata{};
};

RawTrtStatus discover_names(const nvinfer1::ICudaEngine &engine,
                            DiscoveredContract &contract) {
  if (engine.getNbIOTensors() != 2) {
    set_error("raw TensorRT engine must have exactly one input and one output");
    return RAW_TRT_CONTRACT_ERROR;
  }

  for (int32_t index = 0; index < engine.getNbIOTensors(); ++index) {
    const char *name = engine.getIOTensorName(index);
    if (name == nullptr) {
      set_error("TensorRT returned a null I/O tensor name");
      return RAW_TRT_CONTRACT_ERROR;
    }

    switch (engine.getTensorIOMode(name)) {
    case nvinfer1::TensorIOMode::kINPUT:
      if (!contract.input_name.empty()) {
        set_error("raw TensorRT engine has more than one input");
        return RAW_TRT_CONTRACT_ERROR;
      }
      contract.input_name = name;
      break;
    case nvinfer1::TensorIOMode::kOUTPUT:
      if (!contract.output_name.empty()) {
        set_error("raw TensorRT engine has more than one output");
        return RAW_TRT_CONTRACT_ERROR;
      }
      contract.output_name = name;
      break;
    default:
      set_error(std::string("TensorRT tensor '") + name +
                "' has no input/output mode");
      return RAW_TRT_CONTRACT_ERROR;
    }
  }

  if (contract.input_name.empty() || contract.output_name.empty()) {
    set_error("raw TensorRT engine must have one input and one output");
    return RAW_TRT_CONTRACT_ERROR;
  }

  return RAW_TRT_OK;
}

RawTrtStatus validate_shapes(const nvinfer1::ICudaEngine &engine,
                             const RawTrtExpectedContract &expected,
                             DiscoveredContract &contract) {
  const nvinfer1::Dims input = engine.getTensorShape(contract.input_name.c_str());
  const nvinfer1::Dims output = engine.getTensorShape(contract.output_name.c_str());
  if (input.nbDims != 4 || output.nbDims != 2) {
    set_error("raw TensorRT engine requires rank-4 input and rank-2 packed output");
    return RAW_TRT_CONTRACT_ERROR;
  }

  const int64_t actual_input[] = {input.d[1], input.d[2], input.d[3]};
  const int64_t expected_input[] = {expected.channels, expected.height,
                                    expected.width};
  for (size_t index = 0; index < 3; ++index) {
    if (actual_input[index] != expected_input[index] || actual_input[index] <= 0) {
      std::ostringstream message;
      message << "raw TensorRT input shape mismatch: expected [N, "
              << expected.channels << ", " << expected.height << ", "
              << expected.width << "], found [N, " << input.d[1] << ", "
              << input.d[2] << ", " << input.d[3] << "]";
      set_error(message.str());
      return RAW_TRT_CONTRACT_ERROR;
    }
  }

  if (output.d[1] != expected.output_columns || output.d[1] <= 0) {
    std::ostringstream message;
    message << "raw TensorRT output width mismatch: expected "
            << expected.output_columns << ", found " << output.d[1];
    set_error(message.str());
    return RAW_TRT_CONTRACT_ERROR;
  }

  if (input.d[0] != -1 || output.d[0] != -1) {
    set_error("raw TensorRT engine must use a dynamic batch dimension");
    return RAW_TRT_CONTRACT_ERROR;
  }

  contract.metadata.channels = input.d[1];
  contract.metadata.height = input.d[2];
  contract.metadata.width = input.d[3];
  contract.metadata.output_columns = output.d[1];

  return RAW_TRT_OK;
}

RawTrtStatus discover_profile(const nvinfer1::ICudaEngine &engine,
                              const RawTrtExpectedContract &expected,
                              DiscoveredContract &contract) {
  if (engine.getNbOptimizationProfiles() != 1) {
    set_error("raw TensorRT engine must contain exactly one optimization profile");
    return RAW_TRT_CONTRACT_ERROR;
  }

  const nvinfer1::Dims minimum = engine.getProfileShape(
      contract.input_name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
  const nvinfer1::Dims optimum = engine.getProfileShape(
      contract.input_name.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
  const nvinfer1::Dims maximum = engine.getProfileShape(
      contract.input_name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
  if (minimum.nbDims != 4 || optimum.nbDims != 4 || maximum.nbDims != 4) {
    set_error("raw TensorRT engine has an invalid input optimization profile");
    return RAW_TRT_CONTRACT_ERROR;
  }

  const int64_t expected_dimensions[] = {
      expected.channels, expected.height, expected.width};
  for (int32_t axis = 1; axis < 4; ++axis) {
    if (minimum.d[axis] != maximum.d[axis] ||
        minimum.d[axis] != optimum.d[axis] ||
        minimum.d[axis] != expected_dimensions[axis - 1]) {
      set_error("raw TensorRT engine may only vary its batch dimension");
      return RAW_TRT_CONTRACT_ERROR;
    }
  }

  if (minimum.d[0] <= 0 || optimum.d[0] < minimum.d[0] ||
      maximum.d[0] < optimum.d[0] ||
      maximum.d[0] > std::numeric_limits<int32_t>::max()) {
    set_error("raw TensorRT engine has an invalid min/opt/max batch profile");
    return RAW_TRT_CONTRACT_ERROR;
  }

  if (expected.configured_max_batch > 0 &&
      expected.configured_max_batch > maximum.d[0]) {
    std::ostringstream message;
    message << "configured maximum batch " << expected.configured_max_batch
            << " exceeds TensorRT profile maximum " << maximum.d[0];
    set_error(message.str());
    return RAW_TRT_CONTRACT_ERROR;
  }

  contract.metadata.min_batch = minimum.d[0];
  contract.metadata.opt_batch = optimum.d[0];
  contract.metadata.max_batch = maximum.d[0];

  return RAW_TRT_OK;
}

RawTrtStatus discover_contract(const nvinfer1::ICudaEngine &engine,
                               const RawTrtExpectedContract &expected,
                               DiscoveredContract &contract) {
  RawTrtStatus status = discover_names(engine, contract);
  if (status != RAW_TRT_OK) {
    return status;
  }

  status = to_dtype(engine.getTensorDataType(contract.input_name.c_str()),
                    contract.input_name.c_str(), contract.metadata.input_dtype);
  if (status != RAW_TRT_OK) {
    return status;
  }

  status = to_dtype(engine.getTensorDataType(contract.output_name.c_str()),
                    contract.output_name.c_str(), contract.metadata.output_dtype);
  if (status != RAW_TRT_OK) {
    return status;
  }

  status = validate_shapes(engine, expected, contract);
  if (status != RAW_TRT_OK) {
    return status;
  }

  return discover_profile(engine, expected, contract);
}

RawTrtStatus validate_version() {
  const int32_t runtime_version = getInferLibVersion();
  const int32_t runtime_major = runtime_version / 10000;
  const int32_t runtime_minor = (runtime_version / 100) % 100;
  if (runtime_major == NV_TENSORRT_MAJOR && runtime_minor == NV_TENSORRT_MINOR) {
    return RAW_TRT_OK;
  }

  std::ostringstream message;
  message << "TensorRT compile/runtime version mismatch: compiled with "
          << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << ", loaded "
          << runtime_major << '.' << runtime_minor;
  set_error(message.str());
  return RAW_TRT_CONTRACT_ERROR;
}

} // namespace

struct RawTrtSession {
  int32_t device_ordinal;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream = nullptr;
  std::string input_name;
  std::string output_name;
  DeviceBuffer input;
  RawTrtMetadata metadata{};

  ~RawTrtSession() noexcept {
    cudaSetDevice(device_ordinal);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

namespace {

RawTrtStatus create_session(const uint8_t *data, size_t size,
                            const RawTrtExpectedContract &expected,
                            RawTrtSession **out_session,
                            RawTrtMetadata *out_metadata) {
  if (data == nullptr || size == 0 || out_session == nullptr ||
      out_metadata == nullptr || expected.device_ordinal < 0 ||
      expected.channels <= 0 || expected.height <= 0 || expected.width <= 0 ||
      expected.output_columns <= 1 || expected.configured_max_batch < 0) {
    set_error("invalid argument to raw_trt_session_create_from_blob");
    return RAW_TRT_INVALID_ARGUMENT;
  }

  *out_session = nullptr;
  *out_metadata = {};

  RawTrtStatus status = validate_version();
  if (status != RAW_TRT_OK) {
    return status;
  }

  const cudaError_t select_device = cudaSetDevice(expected.device_ordinal);
  if (select_device != cudaSuccess) {
    return cuda_failure("cudaSetDevice(create raw TensorRT session)", select_device);
  }

  auto session = std::make_unique<RawTrtSession>();
  session->device_ordinal = expected.device_ordinal;
  session->runtime.reset(nvinfer1::createInferRuntime(logger()));
  if (!session->runtime) {
    set_error("failed to create TensorRT runtime");
    return RAW_TRT_DESERIALIZE_ERROR;
  }

  session->engine.reset(session->runtime->deserializeCudaEngine(data, size));
  if (!session->engine) {
    set_error("failed to deserialize TensorRT engine; check its TensorRT version and GPU target");
    return RAW_TRT_DESERIALIZE_ERROR;
  }

  DiscoveredContract contract;
  status = discover_contract(*session->engine, expected, contract);
  if (status != RAW_TRT_OK) {
    return status;
  }

  session->context.reset(session->engine->createExecutionContext());
  if (!session->context) {
    set_error("failed to create TensorRT execution context");
    return RAW_TRT_EXECUTION_ERROR;
  }

  const cudaError_t create_stream = cudaStreamCreate(&session->stream);
  if (create_stream != cudaSuccess) {
    return cuda_failure("cudaStreamCreate(raw TensorRT session)", create_stream);
  }

  session->input_name = std::move(contract.input_name);
  session->output_name = std::move(contract.output_name);
  session->metadata = contract.metadata;

  int32_t allocation_batch = expected.configured_max_batch > 0
                                 ? expected.configured_max_batch
                                 : session->metadata.max_batch;
  size_t input_bytes = 0;
  status = input_tensor_bytes(allocation_batch, expected.channels,
                              expected.height, expected.width,
                              session->metadata.input_dtype, input_bytes);
  if (status != RAW_TRT_OK) {
    return status;
  }

  status = session->input.ensure_capacity(input_bytes);
  if (status != RAW_TRT_OK) {
    return status;
  }

  *out_metadata = session->metadata;
  *out_session = session.release();
  return RAW_TRT_OK;
}

RawTrtStatus run_session(RawTrtSession &session, const void *host_input,
                         size_t input_bytes, int32_t batch_size,
                         void *device_output, size_t output_bytes) {
  if (host_input == nullptr || device_output == nullptr || batch_size <= 0) {
    set_error("invalid argument to raw_trt_session_run_host");
    return RAW_TRT_INVALID_ARGUMENT;
  }

  if (batch_size < session.metadata.min_batch ||
      batch_size > session.metadata.max_batch) {
    std::ostringstream message;
    message << "batch " << batch_size << " is outside TensorRT profile ["
            << session.metadata.min_batch << ", " << session.metadata.max_batch
            << ']';
    set_error(message.str());
    return RAW_TRT_CONTRACT_ERROR;
  }

  size_t required_input = 0;
  RawTrtStatus status = input_tensor_bytes(
      batch_size, session.metadata.channels, session.metadata.height,
      session.metadata.width, session.metadata.input_dtype, required_input);
  if (status != RAW_TRT_OK) {
    return status;
  }

  size_t required_output = 0;
  status = tensor_bytes(batch_size, session.metadata.output_columns,
                        session.metadata.output_dtype, required_output);
  if (status != RAW_TRT_OK) {
    return status;
  }

  if (input_bytes < required_input || output_bytes < required_output) {
    std::ostringstream message;
    message << "raw TensorRT buffer too small: input " << input_bytes << "/"
            << required_input << " bytes, output " << output_bytes << "/"
            << required_output << " bytes";
    set_error(message.str());
    return RAW_TRT_INVALID_ARGUMENT;
  }

  const cudaError_t select_device = cudaSetDevice(session.device_ordinal);
  if (select_device != cudaSuccess) {
    return cuda_failure("cudaSetDevice(run raw TensorRT session)", select_device);
  }

  status = session.input.ensure_capacity(required_input);
  if (status != RAW_TRT_OK) {
    return status;
  }

  const nvinfer1::Dims input_shape{4, {batch_size, session.metadata.channels,
                                      session.metadata.height,
                                      session.metadata.width}};
  if (!session.context->setInputShape(session.input_name.c_str(), input_shape)) {
    set_error("TensorRT setInputShape failed for input tensor");
    return RAW_TRT_CONTRACT_ERROR;
  }

  const nvinfer1::Dims output_shape =
      session.context->getTensorShape(session.output_name.c_str());
  if (output_shape.nbDims != 2 || output_shape.d[0] != batch_size ||
      output_shape.d[1] != session.metadata.output_columns) {
    set_error("TensorRT resolved an unexpected packed output shape");
    return RAW_TRT_CONTRACT_ERROR;
  }

  const cudaError_t copy = cudaMemcpyAsync(
      session.input.data(), host_input, required_input, cudaMemcpyHostToDevice,
      session.stream);
  if (copy != cudaSuccess) {
    return cuda_failure("cudaMemcpyAsync(raw TensorRT input)", copy);
  }

  if (!session.context->setTensorAddress(session.input_name.c_str(),
                                         session.input.data()) ||
      !session.context->setTensorAddress(session.output_name.c_str(),
                                         device_output)) {
    set_error("TensorRT setTensorAddress failed");
    return RAW_TRT_EXECUTION_ERROR;
  }

  if (!session.context->enqueueV3(session.stream)) {
    set_error("TensorRT enqueueV3 failed");
    return RAW_TRT_EXECUTION_ERROR;
  }

  const cudaError_t synchronize = cudaStreamSynchronize(session.stream);
  if (synchronize != cudaSuccess) {
    return cuda_failure("cudaStreamSynchronize(raw TensorRT session)",
                        synchronize);
  }

  return RAW_TRT_OK;
}

} // namespace

extern "C" {

RawTrtStatus raw_trt_session_create_from_blob(
    const uint8_t *data, size_t size, const RawTrtExpectedContract *expected,
    RawTrtSession **out_session, RawTrtMetadata *out_metadata) noexcept {
  try {
    if (expected == nullptr) {
      set_error("null expected contract");
      return RAW_TRT_INVALID_ARGUMENT;
    }

    return create_session(data, size, *expected, out_session, out_metadata);
  } catch (const std::exception &error) {
    set_exception_error("raw TensorRT create exception", error.what());
    return RAW_TRT_INTERNAL_ERROR;
  } catch (...) {
    set_error("unknown C++ exception while creating raw TensorRT session");
    return RAW_TRT_INTERNAL_ERROR;
  }
}

RawTrtStatus raw_trt_session_run_host(
    RawTrtSession *session, const void *pinned_host_input, size_t input_bytes,
    int32_t batch_size, void *device_output, size_t output_bytes) noexcept {
  try {
    if (session == nullptr) {
      set_error("null raw TensorRT session");
      return RAW_TRT_INVALID_ARGUMENT;
    }

    return run_session(*session, pinned_host_input, input_bytes, batch_size,
                       device_output, output_bytes);
  } catch (const std::exception &error) {
    set_exception_error("raw TensorRT run exception", error.what());
    return RAW_TRT_INTERNAL_ERROR;
  } catch (...) {
    set_error("unknown C++ exception while running raw TensorRT session");
    return RAW_TRT_INTERNAL_ERROR;
  }
}

void raw_trt_session_destroy(RawTrtSession *session) noexcept {
  try {
    delete session;
  } catch (...) {
    set_error("exception while destroying raw TensorRT session");
  }
}

const char *raw_trt_last_error(void) noexcept {
  return last_error_message.empty() ? "unknown raw TensorRT error"
                                    : last_error_message.c_str();
}

} // extern "C"
