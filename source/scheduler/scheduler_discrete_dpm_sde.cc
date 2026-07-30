/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2026/07/30.
 *
 * DPM-Solver-SDE (2nd order midpoint, stochastic)
 * base on: https://huggingface.co/papers/2211.01095
 *          diffusers DPMSolverSDEScheduler
 *
 * Structure: two UNet evaluations per inference step (same trick as Heun here):
 *   even indices = first-order ancestral half-step to the t-midpoint sigma,
 *   odd indices  = full ancestral step driven by the midpoint x0-prediction.
 * Unlike diffusers (which repeats real sigmas and compensates elsewhere), this
 * implementation stores TRUE midpoint sigmas/timesteps at odd indices, keeping
 * SchedulerBase::scale / time / x0-conversion strictly self-consistent.
 * Noise follows the euler_a in-class RandomGenerator pattern, scaled by
 * random_intensity_ (diffusers s_noise).
 */
#ifndef SCHEDULER_DISCRETE_DPM_SDE
#define SCHEDULER_DISCRETE_DPM_SDE

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class DpmSDEDiscreteScheduler: public SchedulerBase {
private:
    typedef std::vector<float> SdeData;

    static constexpr float SD_SIGMA_FLOOR = 1e-7f;

    RandomGenerator dpm_sde_random;
    SdeData original_sample;                   // sample stored at first-order phase

private:
    static double lambda_at(float sigma_) {    // t = -ln(σ_ratio), sigma floored
        return -std::log(double(std::max(sigma_, SD_SIGMA_FLOOR)));
    }

    // ancestral update shared by both phases:
    // x_to = (σ_down/σ_from)·x + (1-σ_down/σ_from)·x0 + ε·σ_up·intensity
    SdeData ancestral_step(
        const float* x0_data_,
        const float* sample_data_,
        long data_size_,
        double sigma_from_,
        double sigma_to_,
        float random_intensity_
    );

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
    explicit DpmSDEDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
        dpm_sde_random.seed(0);
    }

    ~DpmSDEDiscreteScheduler() override = default;
};

/* Assistant Operations ===================================================*/

DpmSDEDiscreteScheduler::SdeData DpmSDEDiscreteScheduler::ancestral_step(
    const float* x0_data_,
    const float* sample_data_,
    long data_size_,
    double sigma_from_,
    double sigma_to_,
    float random_intensity_
) {
    double sigma_up_ = std::min(
        sigma_to_,
        std::sqrt(sigma_to_ * sigma_to_ * (sigma_from_ * sigma_from_ - sigma_to_ * sigma_to_) /
                  (sigma_from_ * sigma_from_))
    );
    double sigma_down_ = std::sqrt(sigma_to_ * sigma_to_ - sigma_up_ * sigma_up_);
    double f_ = sigma_down_ / sigma_from_;

    SdeData next_samples_(data_size_, 0.0f);
    for (long i = 0; i < data_size_; i++) {
        float noise_ = (sigma_up_ > 0) ? dpm_sde_random.next() * float(sigma_up_) * random_intensity_ : 0.0f;
        next_samples_[i] = float(f_ * double(sample_data_[i]) + (1.0 - f_) * double(x0_data_[i])) + noise_;
    }
    return next_samples_;
}

/* Essential Operations ===================================================*/

// expand each real interval with its t-midpoint: [σ0, σm01, σ1, σm12, ..., σn-1, 0]
uint64_t DpmSDEDiscreteScheduler::correction_steps(uint64_t inference_steps_) {
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

std::vector<float> DpmSDEDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    float random_intensity_
) {
    float sigma_curs = scheduler_sigmas[size_t(step_index_)];
    float sigma_next = scheduler_sigmas[size_t(step_index_ + 1)];   // 0 only at final phase

    // final phase: ancestral step to σ=0 degenerates to the x0-prediction itself
    if (sigma_next <= SD_SIGMA_FLOOR) {
        return SdeData(predict_data_, predict_data_ + data_size_);
    }

    bool first_order_ = (step_index_ % 2 == 0);
    if (first_order_) {
        // half-step σ_i -> σ_mid with the current x0; keep the original sample for phase 2
        original_sample.assign(samples_data_, samples_data_ + data_size_);
        return ancestral_step(predict_data_, samples_data_, data_size_, sigma_curs, sigma_next, random_intensity_);
    } else {
        // full-step σ_i -> σ_{i+1} driven by the midpoint x0 (base converted it at σ_mid)
        double sigma_from_ = scheduler_sigmas[size_t(step_index_ - 1)];
        SdeData next_ = ancestral_step(
            predict_data_, original_sample.data(), data_size_, sigma_from_, sigma_next, random_intensity_
        );
        original_sample.clear();
        return next_;
    }
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_DPM_SDE
