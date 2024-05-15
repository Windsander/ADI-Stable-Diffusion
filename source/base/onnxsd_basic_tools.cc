/*
 * BasicTools
 * Definition: put simple tools we used in here
 * Created by Arikan.Li on 2022/03/11.
 */
#ifndef ONNX_SD_CORE_TOOLS_ONCE
#define ONNX_SD_CORE_TOOLS_ONCE

#include "onnxsd_base_defs.h"

namespace onnx {
namespace sd {
namespace base {
using namespace amon;

#define GET_TENSOR_DATA_SIZE(tensor_shape_, shape_size_) \
    (int) (tensor_shape_[0] * tensor_shape_[1] * tensor_shape_[2] * tensor_shape_[3] * shape_size_)

#define GET_TENSOR_DATA_INFO(tensor_, tensor_data_, tensor_shape_, shape_size_)     \
    auto *tensor_data_ = tensor_.GetTensorData<float>();                            \
    TensorShape tensor_shape_ = tensor_.GetTensorTypeAndShapeInfo().GetShape();     \
    size_t shape_size_ = tensor_.GetTensorTypeAndShapeInfo().GetElementCount()


class RandomGenerator {
protected:
    std::default_random_engine random_generator;
    std::normal_distribution<float> random_style;

public:
    explicit RandomGenerator(float mean_ = 0.0f, float stddev_ = 1.0f) {
        random_style = std::normal_distribution<float>(mean_, stddev_);
    }

    ~RandomGenerator() {
        random_style.reset();
    }

    void seed(uint64_t seed_) {
        if (seed_ == 0) return;
        this->random_generator.seed(seed_);
    }

    float random_at(float mark_) {
        SD_UNUSED(mark_);
        return random_style(random_generator);
    }
};

class TensorHelper {
public:
    static Tensor divide_elements(const Tensor &input_, float denominator_) {
        GET_TENSOR_DATA_INFO(input_, input_data_, input_shape_, input_count_);
        int input_size_ = GET_TENSOR_DATA_SIZE(input_shape_, input_count_);
        float result_data_[input_size_];

        for (int i = 0; i < input_size_; i++) {
            result_data_[i] = input_data_[i] / denominator_;
        }

        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            input_.GetTensorMemoryInfo(), result_data_, input_size_,
            input_shape_.data(), input_shape_.size()
        );

        return result_tensor_;
    }

    static Tensor multiple_elements(const Tensor &input_, float multiplier_) {
        GET_TENSOR_DATA_INFO(input_, input_data_, input_shape_, input_count_);
        int input_size_ = GET_TENSOR_DATA_SIZE(input_shape_, input_count_);
        float result_data_[input_size_];

        for (int i = 0; i < input_size_; i++) {
            result_data_[i] = input_data_[i] * multiplier_;
        }

        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            input_.GetTensorMemoryInfo(), result_data_, input_size_,
            input_shape_.data(), input_shape_.size()
        );

        return result_tensor_;
    }

    static Tensor duplicate(const Tensor &input_, TensorShape shape_) {
        GET_TENSOR_DATA_INFO(input_, input_data_, input_shape_, input_count_);
        int input_size_ = GET_TENSOR_DATA_SIZE(input_shape_, input_count_);
        float result_data_[input_size_];

        for (int i = 0; i < input_size_; i++) {
            result_data_[i] = input_data_[i];
        }

        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
            ), result_data_, input_size_,
            shape_.data(), shape_.size()
        );

        return result_tensor_;
    }

    static Tensor add(const Tensor &input_l_, const Tensor &input_r_, const TensorShape& shape_) {
        GET_TENSOR_DATA_INFO(input_l_, input_data_l_, input_shape_l_, input_count_l_);
        GET_TENSOR_DATA_INFO(input_r_, input_data_r_, input_shape_r_, input_count_r_);
        size_t input_count_l_ = input_l_.GetTensorTypeAndShapeInfo().GetElementCount();
        size_t input_count_r_ = input_l_.GetTensorTypeAndShapeInfo().GetElementCount();
        int input_size_l_ = GET_TENSOR_DATA_SIZE(input_shape_l_, input_count_l_);
        int input_size_r_ = GET_TENSOR_DATA_SIZE(input_shape_r_, input_count_r_);

        if (input_size_l_ != input_size_r_){
            amon_exception(basic_exception(EXC_LOG_ERR, "ERROR:: 2 Tensors adding with data not match"));
        }

        int result_size_ = input_size_l_;
        float result_data_[result_size_];

        for (int i = 0; i < result_size_; i++) {
            result_data_[i] = input_data_l_[i] + input_data_r_[i];
        }

        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            input_l_.GetTensorMemoryInfo(), result_data_, result_size_,
            shape_.data(), shape_.size()
        );

        return result_tensor_;
    }

    static Tensor sub(const Tensor &input_l_, const Tensor &input_r_, const TensorShape& shape_) {
        auto *input_data_l_ = input_l_.GetTensorData<float>();
        auto *input_data_r_ = input_r_.GetTensorData<float>();
        TensorShape input_shape_l_ = input_l_.GetTensorTypeAndShapeInfo().GetShape();
        TensorShape input_shape_r_ = input_l_.GetTensorTypeAndShapeInfo().GetShape();
        size_t input_count_l_ = input_l_.GetTensorTypeAndShapeInfo().GetElementCount();
        size_t input_count_r_ = input_l_.GetTensorTypeAndShapeInfo().GetElementCount();
        int input_size_l_ = GET_TENSOR_DATA_SIZE(input_shape_l_, input_count_l_);
        int input_size_r_ = GET_TENSOR_DATA_SIZE(input_shape_r_, input_count_r_);

        if (input_size_l_ != input_size_r_){
            amon_exception(basic_exception(EXC_LOG_ERR, "ERROR:: 2 Tensors subtract with data not match"));
        }

        int result_size_ = input_size_l_;
        float result_data_[result_size_];

        for (int i = 0; i < result_size_; i++) {
            result_data_[i] = input_data_l_[i] - input_data_r_[i];
        }

        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            input_l_.GetTensorMemoryInfo(), result_data_, result_size_,
            shape_.data(), shape_.size()
        );

        return result_tensor_;
    }

    static Tensor sum(const Tensor* input_tensors_, const int input_size_, const TensorShape& shape_) {
        Tensor result_ = duplicate(input_tensors_[0], shape_);
        for (int i = 1; i < input_size_; ++i) {
            result_ = add(result_, input_tensors_[i], shape_);
        }
        return result_;
    }
};

} // namespace base
} // namespace sd
} // namespace onnx

#endif  // ONNX_SD_CORE_TOOLS_ONCE
