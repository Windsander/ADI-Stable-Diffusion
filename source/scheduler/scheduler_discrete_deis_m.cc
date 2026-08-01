/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2026/07/30.
 *
 * DEIS (multistep, log-rho Lagrange exponential integrator, up to 3rd order)
 * base on: https://huggingface.co/papers/2204.13902
 *          diffusers DEISMultistepScheduler (algorithm_type=deis)
 *
 * Method: history holds genuine eps (m). With rho = sigma (diffusers' rho
 * sigma_t/alpha_t is identical to this framework's sigma), updates in the
 * EDM sample space (x = x0 + sigma*eps) are:
 *   order-1: x_t = x + (sig_t - sig_s0) * m0                     (== DDIM)
 *   order-2: x_t = x + c1*m0 + c2*m1,   c from int of Lagrange basis in log-rho
 *   order-3: x_t = x + c1*m0 + c2*m1 + c3*m2
 * where c_k = ind_fn(sig_t, ...) - ind_fn(sig_s0, ...) with
 *   ind2(t,b,c)   = t*(-ln c + ln t - 1) / (ln b - ln c)
 *   ind3(t,b,c,d) = t*(ln c*(ln d - ln t + 1) - ln d*ln t + ln d + ln^2 t - 2 ln t + 2)
 *                   / ((ln b - ln c)(ln b - ln d))
 *
 * Port notes: warmup ramps order 1->2->3 (diffusers lower_order_nums); final
 * step (sigma_next = 0) always uses order-1, which lands exactly on the x0
 * prediction (log-rho coefficients are singular at rho_t = 0). History is
 * reset per run via correction_steps.
 */
#ifndef SCHEDULER_DISCRETE_DEIS_M
#define SCHEDULER_DISCRETE_DEIS_M

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class DeisMDiscreteScheduler: public SchedulerBase {
private:
    typedef std::vector<float> DeisData;

    static constexpr float SD_SIGMA_FLOOR = 1e-7f;

    std::vector<DeisData> history_dnoise;        // eps history, newest first (cap 3)

private:
    static double floored_(double sigma_) { return std::max(sigma_, double(SD_SIGMA_FLOOR)); }

    // Integrate[(log(t) - log(c)) / (log(b) - log(c)), {t}]
    static double ind2_(double t_, double b_, double c_) {
        double lt_ = std::log(floored_(t_)), lb_ = std::log(floored_(b_)), lc_ = std::log(floored_(c_));
        return t_ * (-lc_ + lt_ - 1.0) / (lb_ - lc_);
    }

    // Integrate[(log(t)-log(c))(log(t)-log(d)) / ((log(b)-log(c))(log(b)-log(d))), {t}]
    static double ind3_(double t_, double b_, double c_, double d_) {
        double lt_ = std::log(floored_(t_)), lb_ = std::log(floored_(b_));
        double lc_ = std::log(floored_(c_)), ld_ = std::log(floored_(d_));
        double num_ = t_ * (lc_ * (ld_ - lt_ + 1.0) - ld_ * lt_ + ld_ + lt_ * lt_ - 2.0 * lt_ + 2.0);
        return num_ / ((lb_ - lc_) * (lb_ - ld_));
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
    explicit DeisMDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
    }

    ~DeisMDiscreteScheduler() override = default;
};

/* Essential Operations ===================================================*/

// base schedule is used as-is; only reset the multistep history per run
uint64_t DeisMDiscreteScheduler::correction_steps(uint64_t inference_steps_) {
    history_dnoise.clear();
    return inference_steps_;
}

std::vector<float> DeisMDiscreteScheduler::execute_method(
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

    DeisData curs_eps_(data_size_, 0.0f);
    for (long i = 0; i < data_size_; i++) {
        curs_eps_[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs_;
    }

    // order ramp: 1 -> 2 -> 3; final step (σ_next = 0) forced order-1
    size_t order_ = std::min<size_t>(size_t(step_index_) + 1, 3);
    if (sigma_next_ <= SD_SIGMA_FLOOR) order_ = 1;
    order_ = std::min(order_, history_dnoise.size() + 1);

    double s_t_  = double(sigma_next_);
    double s_s0_ = double(sigma_curs_);

    std::vector<float> next_samples_(data_size_, 0.0f);
    if (order_ <= 1) {
        // x_t = x + (σ_t − σ_s0)·m0   (== DDIM; at σ_t = 0 lands on x0 exactly)
        double c1_ = s_t_ - s_s0_;
        for (long i = 0; i < data_size_; i++) {
            next_samples_[i] = float(double(samples_data_[i]) + c1_ * double(curs_eps_[i]));
        }
    } else if (order_ == 2) {
        double s_s1_ = double(scheduler_sigmas[size_t(step_index_ - 1)]);
        double c1_ = ind2_(s_t_, s_s0_, s_s1_) - ind2_(s_s0_, s_s0_, s_s1_);
        double c2_ = ind2_(s_t_, s_s1_, s_s0_) - ind2_(s_s0_, s_s1_, s_s0_);
        const DeisData& m1_ = history_dnoise[0];
        for (long i = 0; i < data_size_; i++) {
            next_samples_[i] = float(double(samples_data_[i]) +
                                     c1_ * double(curs_eps_[i]) + c2_ * double(m1_[i]));
        }
    } else {
        double s_s1_ = double(scheduler_sigmas[size_t(step_index_ - 1)]);
        double s_s2_ = double(scheduler_sigmas[size_t(step_index_ - 2)]);
        double c1_ = ind3_(s_t_, s_s0_, s_s1_, s_s2_) - ind3_(s_s0_, s_s0_, s_s1_, s_s2_);
        double c2_ = ind3_(s_t_, s_s1_, s_s2_, s_s0_) - ind3_(s_s0_, s_s1_, s_s2_, s_s0_);
        double c3_ = ind3_(s_t_, s_s2_, s_s0_, s_s1_) - ind3_(s_s0_, s_s2_, s_s0_, s_s1_);
        const DeisData& m1_ = history_dnoise[0];
        const DeisData& m2_ = history_dnoise[1];
        for (long i = 0; i < data_size_; i++) {
            next_samples_[i] = float(double(samples_data_[i]) +
                                     c1_ * double(curs_eps_[i]) + c2_ * double(m1_[i]) + c3_ * double(m2_[i]));
        }
    }

    // record current eps as history (newest first, cap 3)
    history_dnoise.insert(history_dnoise.begin(), std::move(curs_eps_));
    while (history_dnoise.size() > 3) {
        history_dnoise.pop_back();
    }

    return next_samples_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_DEIS_M
