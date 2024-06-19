/*
 * Copyright (c) 2018-2050 SD_UNet - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef ONNX_SD_CORE_EXECUTOR_ONCE
#define ONNX_SD_CORE_EXECUTOR_ONCE

#include "onnxsd_basic_refs.h"

#ifdef ENABLE_TENSOR_RT
#include "tensorrt_provider_factory.h"
#endif
#ifdef ENABLE_CUDA
#include "cuda_provider_factory.h"
#endif
#ifdef ENABLE_COREML
#include "coreml_provider_factory.h"
#endif
#ifdef ENABLE_NNAPI
#include "nnapi_provider_factory.h"
#endif
#include "cpu_provider_factory.h"

namespace onnx {
namespace sd {
namespace base {

using namespace Ort;
using namespace detail;

class ONNXRuntimeExecutor {
private:
    ORTBasicsConfig ort_commons_config = DEFAULT_EXECUTOR_CONFIG;
    OrtOptionConfig ort_session_config;
    int device_id = 0;
    Ort::Env ort_env;

public:
    explicit ONNXRuntimeExecutor(const ORTBasicsConfig &ort_config_ = DEFAULT_EXECUTOR_CONFIG);
    virtual ~ONNXRuntimeExecutor();

    Ort::Session* request_model(const std::string& model_path_);
    Ort::Session* release_model(Ort::Session* model_ptr_);
};

ONNXRuntimeExecutor::ONNXRuntimeExecutor(const ORTBasicsConfig &ort_config_) {
    ort_commons_config = ort_config_;
    ort_env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, DEFAULT_ORT_ENGINE_NAME};
#ifdef ENABLE_TENSOR_RT
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(ort_session_config, device_id));
#endif
#ifdef ENABLE_CUDA
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_config, device_id));
#endif
#ifdef ENABLE_COREML
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(ort_session_config, device_id));
#endif
#ifdef ENABLE_NNAPI
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(ort_session_config, device_id));
#endif
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(ort_session_config, device_id));
    ort_session_config.SetGraphOptimizationLevel(ort_config_.onnx_graph_optimize);
    ort_session_config.SetExecutionMode(ort_config_.onnx_execution_mode);
}

ONNXRuntimeExecutor::~ONNXRuntimeExecutor() {
    ort_env.release();
    ort_session_config.release();
    ort_commons_config = {};
}

Ort::Session* ONNXRuntimeExecutor::request_model(const std::string& model_path_){
    return new Ort::Session(ort_env, model_path_.c_str(), ort_session_config);
}

Ort::Session* ONNXRuntimeExecutor::release_model(Ort::Session* model_ptr_){
    if (model_ptr_){
        model_ptr_->release();
        delete model_ptr_;
    }
    return nullptr;
}

} // namespace base
} // namespace sd
} // namespace onnx

#endif //ONNX_SD_CORE_EXECUTOR_ONCE

