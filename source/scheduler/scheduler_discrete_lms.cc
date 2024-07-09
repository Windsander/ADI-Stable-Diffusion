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
        const float* predict_data_,
        const float* samples_data_,
        long data_size_,
        long step_index_,
        long order_
    ) override;

public:
    explicit LMSDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
    }

    ~LMSDiscreteScheduler() override = default;
};

//python line 135 of scheduling_lms_discrete.py
float LMSDiscreteScheduler::get_lms_coefficient(long history_num_, long t, int h)
{
    // Compute a linear multistep coefficient.
    auto LmsDerivative = [&](float tau)->float {
        float prod = 1.0;
        for (int k = 0; k < history_num_; k++) {
            if (h != k) {
                prod *= (tau - scheduler_sigmas[t - k]) / (scheduler_sigmas[t - h] - scheduler_sigmas[t - k]);
            }
        }
        return prod;
    };

    // Calculate integration with encapsulated IntegralHelper
    int pieces_ = 1000;
    auto integration_ = IntegralHelper::trapezoidal_integral<float>(
        LmsDerivative , scheduler_sigmas[t] , scheduler_sigmas[t + 1], pieces_
    );
    return integration_;
}

std::vector<float> LMSDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    long order_
) {
    std::vector<float> scaled_sample_(data_size_);

    // LMS method:: sigma get
    long history_num = std::min(step_index_ + 1, order_);
    float sigma_curs = scheduler_sigmas[step_index_];

    // LMS method:: current noise decrees
    // 1. Convert to an ODE derivative
    std::vector<float> cur_derivative_(data_size_);
    for (int i = 0; i < data_size_; i++) {
        // derivative_out = (sample - predict_sample) / sigma
        cur_derivative_[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs;
    }

    // 2. Record ODE derivative in history (reverse recs)
    lms_derivatives.insert(lms_derivatives.begin(), cur_derivative_);
    if (lms_derivatives.size() > order_) {
        lms_derivatives.pop_back();
    }

    // 3. compute linear multistep coefficients
    std::vector<float> lms_coeffs_(history_num);
    for (int cur_order_ = 0; cur_order_ < history_num; cur_order_++) {
        // target_coeffs_derivative = lms_method(...)
        lms_coeffs_[cur_order_] = get_lms_coefficient(history_num, step_index_, cur_order_);
    }

    // 4. compute previous sample based on the derivative path
    for (int i = 0; i < data_size_; i++) {
        // output_latent = sample + sum(lms_coeffs * target_coeffs_derivative)
        scaled_sample_[i] = samples_data_[i];
        for (int j = 0; j < history_num; j++) {
            scaled_sample_[i] += lms_coeffs_[j] * lms_derivatives[j][i];
        }
    }

    return scaled_sample_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_LMS
