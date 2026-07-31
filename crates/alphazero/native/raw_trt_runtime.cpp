#include "raw_trt_runtime.h"

#include <NvInfer.h>

#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

namespace {

// TensorRT requires a logger that outlives every object it creates.
class StderrLogger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    // kINFO and kVERBOSE are noisy per-inference-call chatter; only surface
    // warnings and errors so a long-running self-play process does not flood
    // stderr.
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[raw-tensor-rt] %s\n", msg);
    }
  }
};

StderrLogger &logger() {
  static StderrLogger instance;
  return instance;
}

thread_local std::string g_last_error;

void set_error(std::string message) { g_last_error = std::move(message); }

} // namespace

struct RawTrtEngine {
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
};

struct RawTrtContext {
  // Keeps the owning engine alive for as long as any context exists.
  RawTrtEngine *engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  // Non-default stream: enqueueV3(nullptr) makes TensorRT insert extra
  // cudaStreamSynchronize calls and warns on every batch.
  cudaStream_t stream = nullptr;
};

extern "C" {

RawTrtEngine *raw_trt_load_engine(const uint8_t *data, size_t size) {
  if (data == nullptr || size == 0) {
    set_error("engine blob is empty");
    return nullptr;
  }
  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger()));
  if (!runtime) {
    set_error("failed to create TensorRT runtime");
    return nullptr;
  }
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(data, size));
  if (!engine) {
    set_error("failed to deserialize TensorRT engine; it may have been "
              "built for a different TensorRT version or GPU");
    return nullptr;
  }
  auto *out = new RawTrtEngine{std::move(runtime), std::move(engine)};
  return out;
}

void raw_trt_free_engine(RawTrtEngine *engine) { delete engine; }

int32_t raw_trt_num_io_tensors(const RawTrtEngine *engine) {
  if (engine == nullptr) {
    return -1;
  }
  return engine->engine->getNbIOTensors();
}

const char *raw_trt_io_tensor_name(const RawTrtEngine *engine,
                                    int32_t index) {
  if (engine == nullptr) {
    set_error("null engine");
    return nullptr;
  }
  return engine->engine->getIOTensorName(index);
}

int32_t raw_trt_tensor_io_mode(const RawTrtEngine *engine, const char *name) {
  if (engine == nullptr || name == nullptr) {
    return -1;
  }
  return static_cast<int32_t>(engine->engine->getTensorIOMode(name));
}

int32_t raw_trt_tensor_data_type(const RawTrtEngine *engine,
                                  const char *name) {
  if (engine == nullptr || name == nullptr) {
    return -1;
  }
  return static_cast<int32_t>(engine->engine->getTensorDataType(name));
}

RawTrtContext *raw_trt_create_context(RawTrtEngine *engine) {
  if (engine == nullptr) {
    set_error("null engine");
    return nullptr;
  }
  std::unique_ptr<nvinfer1::IExecutionContext> context(
      engine->engine->createExecutionContext());
  if (!context) {
    set_error("failed to create TensorRT execution context");
    return nullptr;
  }
  cudaStream_t stream = nullptr;
  const cudaError_t status = cudaStreamCreate(&stream);
  if (status != cudaSuccess) {
    set_error(std::string("cudaStreamCreate failed: ") +
              cudaGetErrorString(status));
    return nullptr;
  }
  return new RawTrtContext{engine, std::move(context), stream};
}

void raw_trt_free_context(RawTrtContext *context) {
  if (context == nullptr) {
    return;
  }
  if (context->stream != nullptr) {
    cudaStreamDestroy(context->stream);
    context->stream = nullptr;
  }
  delete context;
}

void *raw_trt_context_stream(RawTrtContext *context) {
  if (context == nullptr) {
    set_error("null context");
    return nullptr;
  }
  return context->stream;
}

int32_t raw_trt_set_input_shape(RawTrtContext *context, const char *name,
                                 const int64_t *dims, int32_t nb_dims) {
  if (context == nullptr || name == nullptr || dims == nullptr) {
    set_error("null argument to raw_trt_set_input_shape");
    return -1;
  }
  if (nb_dims <= 0 || nb_dims > nvinfer1::Dims::MAX_DIMS) {
    set_error("invalid dimension count");
    return -1;
  }
  nvinfer1::Dims shape;
  shape.nbDims = nb_dims;
  for (int32_t i = 0; i < nb_dims; ++i) {
    shape.d[i] = dims[i];
  }
  if (!context->context->setInputShape(name, shape)) {
    set_error(std::string("setInputShape failed for tensor ") + name);
    return -1;
  }
  return 0;
}

int32_t raw_trt_set_tensor_address(RawTrtContext *context, const char *name,
                                    void *device_ptr) {
  if (context == nullptr || name == nullptr) {
    set_error("null argument to raw_trt_set_tensor_address");
    return -1;
  }
  if (!context->context->setTensorAddress(name, device_ptr)) {
    set_error(std::string("setTensorAddress failed for tensor ") + name);
    return -1;
  }
  return 0;
}

int32_t raw_trt_get_tensor_shape(RawTrtContext *context, const char *name,
                                  int64_t *dims_out, int32_t *nb_dims_out) {
  if (context == nullptr || name == nullptr || dims_out == nullptr ||
      nb_dims_out == nullptr) {
    set_error("null argument to raw_trt_get_tensor_shape");
    return -1;
  }
  const nvinfer1::Dims shape = context->context->getTensorShape(name);
  if (shape.nbDims < 0) {
    set_error(std::string("shape not yet resolved for tensor ") + name);
    return -1;
  }
  *nb_dims_out = shape.nbDims;
  for (int32_t i = 0; i < shape.nbDims; ++i) {
    dims_out[i] = shape.d[i];
  }
  return 0;
}

int32_t raw_trt_enqueue(RawTrtContext *context, void *stream) {
  if (context == nullptr) {
    set_error("null context");
    return -1;
  }
  if (!context->context->allInputDimensionsSpecified()) {
    set_error("not all input dimensions were specified before enqueue");
    return -1;
  }
  // Null means "use the context's dedicated non-default stream".
  cudaStream_t cuda_stream = stream == nullptr
                                 ? context->stream
                                 : static_cast<cudaStream_t>(stream);
  if (cuda_stream == nullptr) {
    set_error("no CUDA stream available for enqueueV3");
    return -1;
  }
  if (!context->context->enqueueV3(cuda_stream)) {
    set_error("enqueueV3 failed");
    return -1;
  }
  return 0;
}

int32_t raw_trt_synchronize_stream(void *stream) {
  const cudaError_t status =
      cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
  if (status != cudaSuccess) {
    set_error(std::string("cudaStreamSynchronize failed: ") +
              cudaGetErrorString(status));
    return -1;
  }
  return 0;
}

const char *raw_trt_last_error(void) { return g_last_error.c_str(); }

} // extern "C"
