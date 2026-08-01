/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2026/07/30.
 *
 * iPNDM (Improved Pseudo Linear Multistep, 4th order)
 * base on: https://huggingface.co/papers/2202.09778 (formula 12/13, Algorithm 2)
 *
 * Method: Adams-Bashforth extrapolation over the eps history (ets, cap 4):
 *   2: (3e1-e2)/2 | 3: (23e1-16e2+5e3)/12 | 4: (55e1-59e2+37e3-9e4)/24
 * followed by the deterministic DDIM-form update, which in this framework's
 * EDM sample space (x = x0 + σ·ε) is exactly
 *   x_prev = x + (σ_next − σ_cur)·eps_ab      (euler-form with AB-extrapolated eps)
 *
 * Port notes (2026-07-30, evidence in PLAN-supplement.md 1.3 ipndm entry):
 * - diffusers' IPNDMScheduler hardcodes ADM's sin²(πs/2) continuous grid and
 *   feeds UNet timesteps in [0,1); both are invalid for SD models (empirically
 *   verified: mush/garbage output with sd-turbo & sd-v1.5, any spacing).
 *   The paper form (AB4 eps + DDIM update on the model's own σ grid) is what
 *   works for SD, so this port implements the paper form on the framework's
 *   default sigma schedule — same spirit as PLMS in community UIs.
 * - eps is recovered from the base-converted x0 as (sample − x0)/σ_i, so ets
 *   holds genuine eps and v_prediction is handled transparently.
 * - runs entirely on the base schedule: no correction_steps override, karras
 *   sigma strategy composes for free.
 */
#ifndef SCHEDULER_DISCRETE_IPNDM
#define SCHEDULER_DISCRETE_IPNDM

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class IPNDMDiscreteScheduler: public SchedulerBase {
private:
    typedef std::vector<float> IPndmData;

    std::vector<IPndmData> ets_;                 // eps history, oldest-first, cap 4

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
    explicit IPNDMDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
    }

    ~IPNDMDiscreteScheduler() override = default;
};

/* Essential Operations ===================================================*/

// base schedule is used as-is; only reset the multistep history per run
uint64_t IPNDMDiscreteScheduler::correction_steps(uint64_t inference_steps_) {
    ets_.clear();
    return inference_steps_;
}

std::vector<float> IPNDMDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    float random_intensity_
) {
    SD_UNUSED(random_intensity_);

    // recover genuine eps from base-converted x0: eps = (sample - x0) / σ_i
    float sigma_curs_ = scheduler_sigmas[size_t(step_index_)];
    float sigma_next_ = scheduler_sigmas[size_t(step_index_) + 1];

    IPndmData curs_eps_(data_size_, 0.0f);
    for (long i = 0; i < data_size_; i++) {
        curs_eps_[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs_;
    }

    ets_.push_back(std::move(curs_eps_));
    if (ets_.size() > 4) ets_.erase(ets_.begin());

    // Adams-Bashforth extrapolation over eps history (newest-first weights)
    size_t cnt_ = ets_.size();
    IPndmData eps_ab_(ets_.back());
    if (cnt_ == 2) {
        const IPndmData& e1_ = ets_[cnt_ - 1];
        const IPndmData& e2_ = ets_[cnt_ - 2];
        for (long i = 0; i < data_size_; i++) eps_ab_[i] = (3.0f * e1_[i] - e2_[i]) / 2.0f;
    } else if (cnt_ == 3) {
        const IPndmData& e1_ = ets_[cnt_ - 1];
        const IPndmData& e2_ = ets_[cnt_ - 2];
        const IPndmData& e3_ = ets_[cnt_ - 3];
        for (long i = 0; i < data_size_; i++) eps_ab_[i] = (23.0f * e1_[i] - 16.0f * e2_[i] + 5.0f * e3_[i]) / 12.0f;
    } else if (cnt_ >= 4) {
        const IPndmData& e1_ = ets_[cnt_ - 1];
        const IPndmData& e2_ = ets_[cnt_ - 2];
        const IPndmData& e3_ = ets_[cnt_ - 3];
        const IPndmData& e4_ = ets_[cnt_ - 4];
        for (long i = 0; i < data_size_; i++) eps_ab_[i] = (55.0f * e1_[i] - 59.0f * e2_[i] + 37.0f * e3_[i] - 9.0f * e4_[i]) / 24.0f;
    }

    // deterministic DDIM-form update in EDM space
    float sigma_dt_ = sigma_next_ - sigma_curs_;
    std::vector<float> prev_samples_(data_size_, 0.0f);
    for (long i = 0; i < data_size_; i++) {
        prev_samples_[i] = samples_data_[i] + eps_ab_[i] * sigma_dt_;
    }
    return prev_samples_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_IPNDM
