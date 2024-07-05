/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/07/03.
 */
#ifndef SCHEDULER_DISCRETE_HEUN
#define SCHEDULER_DISCRETE_HEUN

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class HeunDiscreteScheduler : public SchedulerBase {
private:
    std::vector<float> prev_derivative;
    std::vector<float> next_predictive;

protected:
    std::vector<float> execute_method(
        const float *predict_data_,
        const float *samples_data_,
        long data_size_,
        long step_index_,
        long order_
    ) override;

public:
    explicit HeunDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_){
    }

    ~HeunDiscreteScheduler() override = default;
};

// base on: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_heun_discrete.py
std::vector<float> HeunDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    long order_
) {
    // There we use order_ to decide, if execute_mth from current step inner next
    bool is_first_order = (order_ > 0);
    bool is_final_round = (step_index_ == scheduler_sigmas.size() - 1);
    std::vector<float> scaled_samples_(data_size_);

    // Heun method:: heun start with euler normal
    float sigma_curs = scheduler_sigmas[step_index_];
    float sigma_next = scheduler_sigmas[step_index_ + 1];
    float sigma_dt = 0;
    {
        sigma_dt = sigma_next - sigma_curs;
    }

    // Heun method derivative logic
    if (is_final_round) {
        // Final round use euler normal to calculate
        for (int i = 0; i < data_size_; i++) {
            scaled_samples_[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs;        // derivative_out = (sample - predict_sample) / sigma
            scaled_samples_[i] = (samples_data_[i] + scaled_samples_[i] * sigma_dt);        // previous_down = sample + derivative_out * dt
        }
    } else if (is_first_order) {
        next_predictive.resize(data_size_);
        prev_derivative.resize(data_size_);

        auto [c_skip_curs, c_out_curs, c_unused_curs] = find_predict_params_at(sigma_curs);
        auto [c_skip_next, c_out_next, c_unused_next] = find_predict_params_at(sigma_next);

        for (int i = 0; i < data_size_; i++) {
            prev_derivative[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs;        // prev_der = (sample - predict_sample) / sigma_curs
            scaled_samples_[i] = (samples_data_[i] + prev_derivative[i] * sigma_dt);        // prev_sample = sample + prev_der * dt
            next_predictive[i] = (predict_data_[i] - samples_data_[i] * c_skip_curs) / c_out_curs * c_out_next; // reverse & get next_predict_data
        }

        scaled_samples_ = std::move(    // needs to be built in local step, order_ recalculate;
            execute_method(scaled_samples_.data(), samples_data_, data_size_, step_index_, -1)
        );

        next_predictive.clear();
        prev_derivative.clear();
    } else {
        for (int i = 0; i < data_size_; i++) {
            scaled_samples_[i] = (predict_data_[i] - next_predictive[i]) / sigma_next;      // curs_der = (prev_sample - predict_next) / sigma_next
            scaled_samples_[i] = (prev_derivative[i] + scaled_samples_[i]) * 0.5f;          // derivative_mid = mean(prev_der, curs_der)
            scaled_samples_[i] = (samples_data_[i] + scaled_samples_[i] * sigma_dt);        // output = sample + derivative_mid * dt
        }
    }

    return scaled_samples_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_HEUN
