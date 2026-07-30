/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2026/07/30.
 *
 * DPM-Solver++ 2S (singlestep, midpoint variant)
 * base on: https://huggingface.co/papers/2211.01095
 *          diffusers DPMSolverSinglestepScheduler (algorithm_type=dpmsolver++, solver_type=midpoint)
 *
 * Structure: two UNet evaluations per inference step (same correction_steps
 * trick as Heun / DPM-SDE here), with TRUE midpoint sigmas/timesteps at odd
 * indices keeping base scale/time/x0-conversion self-consistent:
 *   even indices = order-1 deterministic half-step to the midpoint sigma,
 *   odd indices  = full 2S update over the real interval driven by both x0s:
 *       x_t = (σ_t/σ_A)·x_A + (1-e^{-h})·m1 + 0.5·(1-e^{-h})·(m0-m1)/r0
 *   (EDM translation of diffusers' VP-space update; equivalence verified
 *    numerically for the shared 2M kernel at float32 precision)
 * Final zero-sigma phase degenerates to the x0-prediction (diffusers
 * lower_order_final + final_sigmas_type=zero behavior).
 */
#ifndef SCHEDULER_DISCRETE_DPM_S
#define SCHEDULER_DISCRETE_DPM_S

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class DpmSDiscreteScheduler: public SchedulerBase {
private:
    typedef std::vector<float> DpmData;

    static constexpr float SD_SIGMA_FLOOR = 1e-7f;

    DpmData original_sample;                   // sample at real point A, stored in phase 1
    DpmData first_dnoise;                      // x0-prediction at point A (m0), stored in phase 1

private:
    static double lambda_at(float sigma_) {    // λ = -ln(σ_ratio), sigma floored
        return -std::log(double(std::max(sigma_, SD_SIGMA_FLOOR)));
    }

protected:
    uint64_t correction_steps(uint64_t inference_steps_) override;
    std::vector<float> execute_method(
        const float* predict_data_,
        const float* samples_data_,
        long data_size_,
        long step_index_,
        float random_intensity_
    ) override;

public:
    explicit DpmSDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
    }

    ~DpmSDiscreteScheduler() override = default;
};

/* Essential Operations ===================================================*/

// expand each real interval with its λ-midpoint: [σ0, σm01, σ1, σm12, ..., σn-1, 0]
uint64_t DpmSDiscreteScheduler::correction_steps(uint64_t inference_steps_) {
    std::map<long, int64_t> expanded_timesteps_;
    vector<float> expanded_sigmas_;

    uint64_t real_count_ = uint64_t(scheduler_sigmas.size()) - 1;   // exclude appended 0
    for (uint64_t i = 0; i < real_count_; ++i) {
        double sigma_from_ = scheduler_sigmas[i];
        double sigma_to_   = scheduler_sigmas[i + 1];               // may be 0 at final interval

        expanded_timesteps_.insert(make_pair(long(2 * i), scheduler_timesteps[long(i)]));
        expanded_sigmas_.push_back(float(sigma_from_));

        if (sigma_to_ > double(SD_SIGMA_FLOOR)) {
            double lambda_mid_ = 0.5 * (lambda_at(sigma_from_) + lambda_at(sigma_to_));
            float  sigma_mid_  = float(std::exp(-lambda_mid_));
            expanded_timesteps_.insert(make_pair(long(2 * i + 1), find_timestep_at_sigma(sigma_mid_)));
            expanded_sigmas_.push_back(sigma_mid_);
        }
    }
    expanded_sigmas_.push_back(0);

    scheduler_timesteps = expanded_timesteps_;
    scheduler_sigmas    = expanded_sigmas_;
    return inference_steps_ * 2 - 1;
}

std::vector<float> DpmSDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    float random_intensity_
) {
    SD_UNUSED(random_intensity_);

    float sigma_curs = scheduler_sigmas[size_t(step_index_)];
    float sigma_next = scheduler_sigmas[size_t(step_index_ + 1)];   // 0 only at final phase

    // final phase: order-1 degenerates to the x0-prediction itself
    if (sigma_next <= SD_SIGMA_FLOOR) {
        return DpmData(predict_data_, predict_data_ + data_size_);
    }

    bool first_order_ = (step_index_ % 2 == 0);
    std::vector<float> next_samples_(data_size_, 0.0f);

    if (first_order_) {
        // phase 1: order-1 half-step σ_A -> σ_mid (deterministic);
        // keep x_A and m0 for the 2S correction in phase 2
        double h_half_ = lambda_at(sigma_next) - lambda_at(sigma_curs);
        double e_neg_h_ = std::exp(-h_half_);
        double f_ = double(sigma_next) / double(sigma_curs);
        for (long i = 0; i < data_size_; i++) {
            next_samples_[i] = float(f_ * double(samples_data_[i]) +
                                     (1.0 - e_neg_h_) * double(predict_data_[i]));
        }
        original_sample.assign(samples_data_, samples_data_ + data_size_);
        first_dnoise.assign(predict_data_, predict_data_ + data_size_);
    } else {
        // phase 2: full 2S step σ_A -> σ_B with midpoint x0 (m1, converted by base at σ_mid)
        // x_t = (σ_B/σ_A)·x_A + (1-e^{-h})·m1 + 0.5·(1-e^{-h})·(m0-m1)/r0
        double sigma_from_ = scheduler_sigmas[size_t(step_index_ - 1)];
        double h_     = lambda_at(sigma_next) - lambda_at(sigma_from_);     // full interval
        double h_0_   = lambda_at(sigma_curs) - lambda_at(sigma_from_);     // A -> midpoint
        double r0_    = h_0_ / h_;
        double e_neg_h_ = std::exp(-h_);
        double f_     = double(sigma_next) / sigma_from_;
        double c_d1_  = 0.5 * (1.0 - e_neg_h_) / r0_;
        for (long i = 0; i < data_size_; i++) {
            double m1_ = double(predict_data_[i]);
            double d1_ = double(first_dnoise[i]) - m1_;
            next_samples_[i] = float(f_ * double(original_sample[i]) +
                                     (1.0 - e_neg_h_) * m1_ + c_d1_ * d1_);
        }
        original_sample.clear();
        first_dnoise.clear();
    }

    return next_samples_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_DPM_S
