/*
 * Copyright (c) 2018-2050 SD_UNet - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef ONNX_SD_CORE_EXECUTOR_ONCE
#define ONNX_SD_CORE_EXECUTOR_ONCE

#include "onnxsd_base_defs.h"
#include "onnxsd_basic_core_config.cc"

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
    ort_env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "OrtSD-Engine"};
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(ort_session_config, device_id));
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_config, device_id));
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

