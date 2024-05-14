/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_DISCRETE_LMS
#define SCHEDULER_DISCRETE_LMS

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class LMSDiscreteScheduler: public SchedulerBase {
private:
    std::vector<std::vector<float>> lms_derivatives;

private:
    float get_lms_coefficient(long order, long t, int current_order);

protected:
    std::vector<float> execute_method(
        const float* samples_data_,
        const float* predict_data_,
        int elements_in_batch,
        long step_index_,
        long order_
    ) override;

public:
    explicit LMSDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
    }

    ~LMSDiscreteScheduler() override = default;
};

//python line 135 of scheduling_lms_discrete.py
float LMSDiscreteScheduler::get_lms_coefficient(long order, long t, int current_order)
{
    // Compute a linear multistep coefficient.

    auto LmsDerivative = [&](float tau)->float {
        float prod = 1.0;
        for (int k = 0; k < order; k++) {
            if (current_order == k) {
                continue;
            }
            prod *= (tau - scheduler_sigmas[t - k]) / (scheduler_sigmas[t - current_order] - scheduler_sigmas[t - k]);
        }
        return prod;
    };

    float integrated_coeff = std::accumulate(
        scheduler_sigmas.begin() + t, scheduler_sigmas.begin() + t + 2, 1e-4,
        [&](float acc, float sigma) -> float {
            return acc + LmsDerivative(sigma);
        }
    );
    return integrated_coeff;
}

std::vector<float> LMSDiscreteScheduler::execute_method(
    const float* samples_data_,
    const float* predict_data_,
    int elements_in_batch,
    long step_index_,
    long order_
) {
    std::vector<float> scaled_sample_(elements_in_batch);

    // LMS method:: sigma get
    float sigma_curs = scheduler_sigmas[step_index_];

    // LMS method:: current noise decrees
    // 1. Convert to an ODE derivative
    std::vector<float> cur_derivative_(elements_in_batch);
    for (int i = 0; i < elements_in_batch; i++) {
        // derivative_out = (sample - predict_sample) / sigma
        cur_derivative_[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs;
    }

    // 2. Record ODE derivative in history (reverse recs)
    lms_derivatives.insert(lms_derivatives.begin(), cur_derivative_);
    if (lms_derivatives.size()>order_){
        lms_derivatives.pop_back();
    }

    // 3. compute linear multistep coefficients
    long remark_order_ = std::min(step_index_ + 1, order_);
    std::vector<float> lms_coeffs_(remark_order_);
    for (int cur_order = 0; cur_order < remark_order_; cur_order++) {
        // target_coeffs_derivative = lms_method(...)
        lms_coeffs_[cur_order] = get_lms_coefficient(remark_order_, step_index_, cur_order);
    }

    // 4. compute previous sample based on the derivative path
    std::vector<std::vector<float>> lms_der_products_;
    for (int i = 0; i < lms_coeffs_.size(); i++) {
        // history_product_recs = sample + lms_coeffs * target_coeffs_derivative
        const std::vector<float>& derivative_ = lms_derivatives[lms_derivatives.size() - 1 - i];
        for (int j = 0; j < derivative_.size(); j++) {
            lms_der_products_[i][j] = samples_data_[j] + lms_coeffs_[i] * derivative_[j];
        }
    }

    for (const auto & product_ : lms_der_products_) {
        // sum_product_tensor = sum(history_product_recs)
        for (int j = 0; j < product_.size(); j++) {
            scaled_sample_[j] += product_[j];
        }
    }
    for (int i = 0; i < scaled_sample_.size(); i++) {
        // output_latent = sample + sum_product_tensor
        scaled_sample_[i] += samples_data_[i];
    }

    return scaled_sample_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_LMS
