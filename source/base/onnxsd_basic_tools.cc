/*
 * BasicTools
 * Definition: put simple tools we used in here
 * Created by Arikan.Li on 2022/03/11.
 */
#ifndef ONNX_SD_CORE_TOOLS_ONCE
#define ONNX_SD_CORE_TOOLS_ONCE

#include "onnxsd_base_defs.h"
#include "onnxsd_basic_core_config.cc"

namespace onnx {
namespace sd {
namespace base {
using namespace amon;

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

    float next() {
        float u1 = random_style(random_generator);
        float u2 = random_style(random_generator);
        float radius = std::sqrt(-2.0f * std::log(u1));
        float theta = float(2.0f * M_PI) * u2;
        float standard = radius * std::cos(theta);
        return standard;
    }
};

class TensorHelper {

#define GET_TENSOR_DATA_SIZE(tensor_shape_, shape_size_) \
    [&]()->long{                                         \
        int64_t data_size_ = 1;                          \
        for (long long i: tensor_shape_) {               \
            data_size_ = data_size_ * i;                 \
        }                                                \
        return (long)(data_size_ * shape_size_);         \
    }()

#define GET_TENSOR_DATA_INFO(tensor_, tensor_data_, tensor_shape_, shape_size_)     \
    auto *tensor_data_ = tensor_.GetTensorData<float>();                            \
    TensorShape tensor_shape_ = tensor_.GetTensorTypeAndShapeInfo().GetShape();     \
    size_t shape_size_ = tensor_.GetTensorTypeAndShapeInfo().GetElementCount()

public:
    static long get_data_size(const Tensor &input_) {
        TensorShape tensor_shape_ = input_.GetTensorTypeAndShapeInfo().GetShape();
        size_t input_count_ = input_.GetTensorTypeAndShapeInfo().GetElementCount();
        long input_size_ = GET_TENSOR_DATA_SIZE(tensor_shape_, input_count_);
        return input_size_;
    }

    template<class T>
    static Tensor create(TensorShape shape_, vector<T> value_) {
        Tensor result_tensor_ = Tensor::CreateTensor<T>(
            Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
            ), value_.data(), value_.size(),
            shape_.data(), shape_.size()
        );

        return result_tensor_;
    }

    static Tensor random(TensorShape shape_, RandomGenerator random_, float factor_) {
        long input_size_ = GET_TENSOR_DATA_SIZE(shape_, 1);
        float result_data_[input_size_];

        for (int i = 0; i < input_size_; i++) {
            result_data_[i] = random_.next() * factor_;
        }

        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
            ), result_data_, input_size_,
            shape_.data(), shape_.size()
        );

        return result_tensor_;
    }

    static Tensor divide(const Tensor &input_, float denominator_, float offset_ = 0.0f) {
        GET_TENSOR_DATA_INFO(input_, input_data_, input_shape_, input_count_);
        long input_size_ = GET_TENSOR_DATA_SIZE(input_shape_, input_count_);
        float result_data_[input_size_];

        for (int i = 0; i < input_size_; i++) {
            result_data_[i] = input_data_[i] / denominator_ + offset_;
        }

        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            input_.GetTensorMemoryInfo(), result_data_, input_size_,
            input_shape_.data(), input_shape_.size()
        );

        return result_tensor_;
    }

    static Tensor multiple(const Tensor &input_, float multiplier_, float offset_ = 0.0f) {
        GET_TENSOR_DATA_INFO(input_, input_data_, input_shape_, input_count_);
        long input_size_ = GET_TENSOR_DATA_SIZE(input_shape_, input_count_);
        float result_data_[input_size_];

        for (int i = 0; i < input_size_; i++) {
            result_data_[i] = input_data_[i] * multiplier_ + offset_;
        }

        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            input_.GetTensorMemoryInfo(), result_data_, input_size_,
            input_shape_.data(), input_shape_.size()
        );

        return result_tensor_;
    }

    static Tensor duplicate(const Tensor &input_, TensorShape shape_) {
        GET_TENSOR_DATA_INFO(input_, input_data_, input_shape_, input_count_);
        long input_size_ = GET_TENSOR_DATA_SIZE(input_shape_, input_count_);
        float result_data_[input_size_];

        for (int i = 0; i < input_size_; i++) {
            result_data_[i] = input_data_[i];
        }

        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            input_.GetTensorMemoryInfo(), result_data_, input_size_,
            shape_.data(), shape_.size()
        );

        return result_tensor_;
    }

    static Tensor split(const Tensor &input_) {
        GET_TENSOR_DATA_INFO(input_, input_data_, input_shape_, input_count_);
        long input_size_ = GET_TENSOR_DATA_SIZE(input_shape_, input_count_);
        float result_data_[input_size_];

        for (int i = 0; i < input_size_; i++) {
            result_data_[i] = input_data_[i];
        }

        TensorShape shape_ = input_shape_;
        shape_[0] = 1;
        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            input_.GetTensorMemoryInfo(), result_data_, input_size_,
            shape_.data(), shape_.size()
        );

        return result_tensor_;
    }

    static Tensor guidance(const Tensor &input_l_, const Tensor &input_r_, float guidance_scale_) {
        GET_TENSOR_DATA_INFO(input_l_, input_data_l_, input_shape_l_, input_count_l_);
        GET_TENSOR_DATA_INFO(input_r_, input_data_r_, input_shape_r_, input_count_r_);
        long input_size_l_ = GET_TENSOR_DATA_SIZE(input_shape_l_, input_count_l_);
        long input_size_r_ = GET_TENSOR_DATA_SIZE(input_shape_r_, input_count_r_);

        if (input_size_l_ != input_size_r_){
            amon_exception(basic_exception(EXC_LOG_ERR, "ERROR:: 2 Tensors guidance without match"));
        }

        TensorShape result_shape_ = input_shape_l_;
        long result_size_ = input_size_l_;
        float result_data_[result_size_];

        for (int i = 0; i < result_size_; i++) {
            result_data_[i] = input_data_l_[i] + guidance_scale_ * (input_data_r_[i] - input_data_l_[i]);
        }

        Tensor result_tensor_ = Tensor::CreateTensor<float>(
            input_l_.GetTensorMemoryInfo(), result_data_, result_size_,
            result_shape_.data(), result_shape_.size()
        );

        return result_tensor_;
    }

    static Tensor add(const Tensor &input_l_, const Tensor &input_r_, const TensorShape& shape_) {
        GET_TENSOR_DATA_INFO(input_l_, input_data_l_, input_shape_l_, input_count_l_);
        GET_TENSOR_DATA_INFO(input_r_, input_data_r_, input_shape_r_, input_count_r_);
        long input_size_l_ = GET_TENSOR_DATA_SIZE(input_shape_l_, input_count_l_);
        long input_size_r_ = GET_TENSOR_DATA_SIZE(input_shape_r_, input_count_r_);

        if (input_size_l_ != input_size_r_){
            amon_exception(basic_exception(EXC_LOG_ERR, "ERROR:: 2 Tensors adding with data not match"));
        }

        long result_size_ = input_size_l_;
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
        GET_TENSOR_DATA_INFO(input_l_, input_data_l_, input_shape_l_, input_count_l_);
        GET_TENSOR_DATA_INFO(input_r_, input_data_r_, input_shape_r_, input_count_r_);
        long input_size_l_ = GET_TENSOR_DATA_SIZE(input_shape_l_, input_count_l_);
        long input_size_r_ = GET_TENSOR_DATA_SIZE(input_shape_r_, input_count_r_);

        if (input_size_l_ != input_size_r_){
            amon_exception(basic_exception(EXC_LOG_ERR, "ERROR:: 2 Tensors subtract with data not match"));
        }

        long result_size_ = input_size_l_;
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

    static Tensor sum(const Tensor* input_tensors_, const long input_size_, const TensorShape& shape_) {
        Tensor result_ = duplicate(input_tensors_[0], shape_);
        for (int i = 1; i < input_size_; ++i) {
            result_ = add(result_, input_tensors_[i], shape_);
        }
        return result_;
    }

#undef GET_TENSOR_DATA_INFO
#undef GET_TENSOR_DATA_SIZE
};

} // namespace base
} // namespace sd
} // namespace onnx

#endif  // ONNX_SD_CORE_TOOLS_ONCE
