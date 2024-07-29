/*
 * Copyright (c) 2018-2050 SD_ModelBase - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_BASE_H
#define MODEL_BASE_H

#include <utility>

#include "onnxsd_foundation.cc"

namespace onnx {
namespace sd {
namespace units {

using namespace base;
using namespace amon;
using namespace Ort;
using namespace detail;

class ModelBase {
private:
    typedef std::string OrtMdlPath;
    typedef struct OrtMdlMeta {
        std::vector<std::string> tensor_names_i{};
        std::vector<std::string> tensor_names_o{};
        size_t tensor_count_i = 0;
        size_t tensor_count_o = 0;
    } OrtMdlMeta;

private:
    OrtSession model_session = nullptr;
    OrtMdlPath model_path;
    OrtMdlMeta model_meta{};

protected:
    void print_model_detail(const Ort::AllocatorWithDefaultOptions& allocator, bool is_input);
    void execute(std::vector<Tensor>& input_tensors_, std::vector<Tensor>& output_tensors_);

protected:
    virtual void generate_output(std::vector<Tensor>& output_tensors_) = 0;

public:
    explicit ModelBase(std::string model_path_) : model_path(std::move(model_path_)) {};
    virtual ~ModelBase() = default;

    void init(ONNXRuntimeExecutor &ort_executor_);
    void release(ONNXRuntimeExecutor &ort_executor_);
};

void ModelBase::print_model_detail(const Ort::AllocatorWithDefaultOptions& allocator, bool is_input) {
    size_t num_nodes = is_input ? model_session->GetInputCount() : model_session->GetOutputCount();
    std::cout << (is_input ? "Input" : "Output")  << " [" << std::endl;

    for (size_t i = 0; i < num_nodes; ++i) {
        Ort::AllocatedStringPtr name = (
            (is_input) ?
            model_session->GetInputNameAllocated(i, allocator) :
            model_session->GetOutputNameAllocated(i, allocator)
        );
        Ort::TypeInfo type_info = (
            (is_input) ?
            model_session->GetInputTypeInfo(i) :
            model_session->GetOutputTypeInfo(i)
        );
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();

        std::cout << "  " << name.get() << " {" << std::endl;
        std::cout << "    Type : " << TensorHelper::get_tensor_type(type).c_str() << std::endl;
        std::cout << "    Shape: ";
        std::vector<int64_t> node_dims = tensor_info.GetShape();
        for (size_t j = 0; j < node_dims.size(); ++j) {
            if (node_dims[j] == -1) {
                std::cout << "Dynamic";
            } else {
                std::cout << node_dims[j];
            }
            if (j < node_dims.size() - 1) std::cout << " x ";
        }
        std::cout << std::endl;
        std::cout  << "  }, " << std::endl;
    }
    std::cout << "]" << std::endl;
}

void ModelBase::init(ONNXRuntimeExecutor &ort_executor_) {
    if (model_path.empty()) {
        amon_report(class_exception(EXC_LOG_ERR, "ERROR:: model path is NaN"));
        return;
    }
    model_session = ort_executor_.request_model(model_path);
    if (!model_session) {
        amon_report(class_exception(EXC_LOG_ERR, "ERROR:: model create failed"));
        return;
    }

    size_t input_count = model_session->GetInputCount();
    size_t output_count = model_session->GetOutputCount();

    Ort::AllocatorWithDefaultOptions ort_alloc;
    for (int i = 0; i < input_count; i++) {
        auto input_name = model_session->GetInputNameAllocated(i, ort_alloc);
        model_meta.tensor_names_i.emplace_back(input_name.get());
    }
    for (int i = 0; i < output_count; i++) {
        auto input_name = model_session->GetOutputNameAllocated(i, ort_alloc);
        model_meta.tensor_names_o.emplace_back(input_name.get());
    }

    model_meta.tensor_count_i = input_count;
    model_meta.tensor_count_o = output_count;

    std::cout << model_path.c_str() << std::endl;
    print_model_detail(ort_alloc, true);
    print_model_detail(ort_alloc, false);
}

void ModelBase::execute(std::vector<Tensor>& input_tensors_, std::vector<Tensor>& output_tensors_) {
    if (!model_session) amon_report(class_exception(EXC_LOG_ERR, "ERROR:: model not found"));
    try {
        Ort::IoBinding io_binding(*model_session);
        for (size_t i = 0; i < model_meta.tensor_count_i; ++i) {
            io_binding.BindInput(model_meta.tensor_names_i[i].c_str(), input_tensors_[i]);
        }
        for (size_t i = 0; i < model_meta.tensor_count_o; ++i) {
            io_binding.BindOutput(model_meta.tensor_names_o[i].c_str(), output_tensors_[i]);
        }
        model_session->Run(Ort::RunOptions{nullptr}, io_binding);
    } catch (const Ort::Exception &e) {
        std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }
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

