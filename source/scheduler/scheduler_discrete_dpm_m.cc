/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2026/07/30.
 *
 * DPM-Solver++ 2M (multistep, midpoint variant)
 * base on: https://huggingface.co/papers/2211.01095
 *          diffusers DPMSolverMultistepScheduler (algorithm_type=dpmsolver++, solver_type=midpoint)
 *
 * Note: formulas are translated to this project's EDM sample space (x = x0 + σ·ε,
 * see SchedulerBase::scale / find_predict_params_at), equivalent to diffusers'
 * VP-space update after coordinate mapping x_vp = α·x_edm (verified numerically
 * against diffusers 0.39 to float32 precision).
 * Final step (σ_next = 0) uses order-1, which degenerates to the x0-prediction
 * itself (same as diffusers lower_order_final with zero sigma).
 */
#ifndef SCHEDULER_DISCRETE_DPM_M
#define SCHEDULER_DISCRETE_DPM_M

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class DpmMDiscreteScheduler: public SchedulerBase {
private:
    typedef std::vector<float> DpmData;

    static constexpr float SD_SIGMA_FLOOR = 1e-7f;

    std::vector<DpmData> history_dnoise;       // model x0-predictions, newest first (cap 2)

private:
    static double lambda_at(float sigma_) {    // λ = -ln(σ_ratio), sigma floored
        return -std::log(double(std::max(sigma_, SD_SIGMA_FLOOR)));
    }

protected:
    std::vector<float> execute_method(
        const float* predict_data_,
        const float* samples_data_,
        long data_size_,
        long step_index_,
        float random_intensity_
    ) override;

public:
    explicit DpmMDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
    }

    ~DpmMDiscreteScheduler() override = default;
};

/* Essential Operations ===================================================*/

std::vector<float> DpmMDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    float random_intensity_
) {
    SD_UNUSED(random_intensity_);

    // predict_data_ is already the x0-prediction (converted by base with c_skip/c_out)
    DpmData curs_dnoised_(predict_data_, predict_data_ + data_size_);

    float sigma_curs = scheduler_sigmas[size_t(step_index_)];
    float sigma_next = scheduler_sigmas[size_t(step_index_ + 1)];   // appended 0 at final step

    // final step: order-1 degenerates to the x0-prediction (diffusers lower_order_final + zero sigma)
    if (sigma_next <= SD_SIGMA_FLOOR) {
        return curs_dnoised_;
    }

    double lambda_s0 = lambda_at(sigma_curs);
    double h_        = lambda_at(sigma_next) - lambda_s0;            // > 0
    double e_neg_h_  = std::exp(-h_);
    double f_        = double(sigma_next) / double(sigma_curs);

    // order: 2M needs one history entry; warmup step falls back to order-1 (DDIM)
    bool second_order_ = (step_index_ > 0) && !history_dnoise.empty() &&
                         (scheduler_config.scheduler_maintain_cache > 1);

    std::vector<float> next_samples_(data_size_, 0.0f);
    if (!second_order_) {
        // x_t = (σ_t/σ_s) * x + (1-e^{-h}) * m0
        for (long i = 0; i < data_size_; i++) {
            next_samples_[i] = float(f_ * double(samples_data_[i]) + (1.0 - e_neg_h_) * double(curs_dnoised_[i]));
        }
    } else {
        // x_t = (σ_t/σ_s) * x + (1-e^{-h}) * m0 + 0.5 * (1-e^{-h}) * D1
        // D1 = (m0 - m1) / r0, r0 = h_0 / h, h_0 = λ_s0 - λ_s1
        double lambda_s1 = lambda_at(scheduler_sigmas[size_t(step_index_ - 1)]);
        double h_0_      = lambda_s0 - lambda_s1;
        double r0_       = h_0_ / h_;
        double c_d1_     = 0.5 * (1.0 - e_neg_h_) / r0_;
        for (long i = 0; i < data_size_; i++) {
            double m0_ = double(curs_dnoised_[i]);
            double d1_ = m0_ - double(history_dnoise[0][i]);
            next_samples_[i] = float(f_ * double(samples_data_[i]) +
                                     (1.0 - e_neg_h_) * m0_ + c_d1_ * d1_);
        }
    }

    // record current model output as next step's m1 (2M only needs the latest one)
    history_dnoise.insert(history_dnoise.begin(), std::move(curs_dnoised_));
    while (history_dnoise.size() > 2) {
        history_dnoise.pop_back();
    }

    return next_samples_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_DPM_M
