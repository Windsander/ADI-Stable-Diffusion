/*
 * Copyright (c) 2018-2050 SD_ModelBase - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_BASE_H
#define MODEL_BASE_H

#include "onnxsd_foundation.cc"

namespace onnx {
namespace sd {
namespace units {

using namespace base;
using namespace amon;
using namespace scheduler;
using namespace Ort;
using namespace detail;

class ModelBase {
private:
    typedef std::string OrtMdlPath;
    typedef struct OrtMdlMeta {
        std::vector<const char*> tensor_names_i{};
        std::vector<const char*> tensor_names_o{};
        size_t tensor_count_i = 0;
        size_t tensor_count_o = 0;
    } OrtMdlMeta;

private:
    OrtSession model_session = nullptr;
    OrtMdlPath model_path;
    OrtMdlMeta model_meta{};

protected:
    void execute(std::vector<Tensor>& input_tensors_, std::vector<Tensor>& output_tensors_);

public:
    explicit ModelBase(const std::string &model_path_) : model_path(model_path_) {};
    virtual ~ModelBase() = default;

    void init(ONNXRuntimeExecutor &ort_executor_);
    void release(ONNXRuntimeExecutor &ort_executor_);
};

void ModelBase::init(ONNXRuntimeExecutor &ort_executor_) {
    if (model_path.empty()) amon_report(class_exception(EXC_LOG_ERR, "ERROR:: model path is NaN"));
    model_session = ort_executor_.request_model(model_path);
    if (!model_session) amon_report(class_exception(EXC_LOG_ERR, "ERROR:: model create failed"));

    size_t input_count = model_session->GetInputCount();
    size_t output_count = model_session->GetOutputCount();

    Ort::AllocatorWithDefaultOptions ort_alloc;
    for (int i = 0; i < input_count; i++) {
        auto input_name = model_session->GetInputNameAllocated(i, ort_alloc);
        model_meta.tensor_names_i.push_back(input_name.get());
    }
    for (int i = 0; i < output_count; i++) {
        auto input_name = model_session->GetOutputNameAllocated(i, ort_alloc);
        model_meta.tensor_names_o.push_back(input_name.get());
    }
    model_meta.tensor_count_i = input_count;
    model_meta.tensor_count_o = output_count;
}

void ModelBase::execute(std::vector<Tensor>& input_tensors_, std::vector<Tensor>& output_tensors_) {
    if (!model_session) amon_report(class_exception(EXC_LOG_ERR, "ERROR:: model not found"));
    model_session->Run(
        Ort::RunOptions{nullptr},
        model_meta.tensor_names_i.data(), input_tensors_.data(), model_meta.tensor_count_i,
        model_meta.tensor_names_o.data(), output_tensors_.data(), model_meta.tensor_count_o
    );
}

void ModelBase::release(ONNXRuntimeExecutor &ort_executor_) {
    ort_executor_.release_model(model_session);
    model_session = nullptr;
    model_path.clear();
    model_meta.~OrtMdlMeta();
}

} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_BASE_H

