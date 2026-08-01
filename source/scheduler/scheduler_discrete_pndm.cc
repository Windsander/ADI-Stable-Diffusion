/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2026/07/30.
 *
 * PNDM (Pseudo Numerical Diffusion Model, prk + plms)
 * base on: https://huggingface.co/papers/2202.09778
 *          diffusers PNDMScheduler (skip_prk_steps=false, timestep_spacing=leading)
 *
 * Structure: first 4*(k-1) internal steps are 4-eval Runge-Kutta quarters
 * (weights 1/6, 1/3, 1/3, 1/6), remaining steps are 1-eval PLMS with
 * Adams-Bashforth extrapolation over the eps history (ets):
 *   2: (3e1-e2)/2 | 3: (23e1-16e2+5e3)/12 | 4: (55e1-59e2+37e3-9e4)/24
 * then the deterministic DDIM-style update.
 *
 * Port notes: eps is recovered from the base-converted x0 as (sample - x0)/σ_i,
 * so ets always holds genuine eps and v_prediction is handled transparently.
 * Internal timestep sequence (leading spacing + prk quarters) is rebuilt in
 * correction_steps.
 *
 * Coordinate fix (2026-07-30): paper formula (9) is derived for VP-space
 * samples, but this framework carries EDM samples (x = x0 + σ·ε). Algebra:
 *   prev_vp = √(α_p/α_t)·x_vp + [√(1-α_p) − √(α_p(1-α_t)/α_t)]·eps
 * with x_vp = √α·x_edm collapses to the EDM update
 *   x_prev = x + (σ_prev − σ_ref)·eps        (σ = √((1-ᾱ)/ᾱ))
 * i.e. euler's update with the RK/AB-extrapolated eps. The first version of
 * this port applied the alpha-space formula directly to EDM samples, which
 * broke the σ geometry and produced pure noise at every step count.
 */
#ifndef SCHEDULER_DISCRETE_PNDM
#define SCHEDULER_DISCRETE_PNDM

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class PNDMDiscreteScheduler: public SchedulerBase {
private:
    typedef std::vector<float> PndmData;

    std::vector<PndmData> ets_;                // eps history, oldest-first, cap 4
    PndmData cur_model_output_;                // RK accumulator
    PndmData cur_sample_;                      // RK anchor sample
    long pndm_counter_ = 0;
    long prk_total_steps_ = 0;                 // = 4 * (k-1), k = min(4, inference_steps)
    long delta_ = 0;                           // training_steps / inference_steps
    long delta_half_ = 0;
    std::vector<int64_t> prk_anchor_ts_;       // prk_timesteps[(c//4)*4] per RK group

private:
    // deterministic update in EDM space: x_prev = x + (σ_prev − σ_ref)·eps
    PndmData get_prev_sample(
        const float* eps_data_,
        const float* sample_data_,
        long data_size_,
        float sigma_ref_,
        float sigma_prev_
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
    explicit PNDMDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
    }

    ~PNDMDiscreteScheduler() override = default;
};

/* Assistant Operations ===================================================*/

PNDMDiscreteScheduler::PndmData PNDMDiscreteScheduler::get_prev_sample(
    const float* eps_data_,
    const float* sample_data_,
    long data_size_,
    float sigma_ref_,
    float sigma_prev_
) {
    float sigma_dt_ = sigma_prev_ - sigma_ref_;
    PndmData prev_samples_(data_size_, 0.0f);
    for (long i = 0; i < data_size_; i++) {
        prev_samples_[i] = sample_data_[i] + eps_data_[i] * sigma_dt_;
    }
    return prev_samples_;
}

/* Essential Operations ===================================================*/

// rebuild internal sequence: [prk quarters (leading spacing), plms descending]
uint64_t PNDMDiscreteScheduler::correction_steps(uint64_t inference_steps_) {
    delta_      = long(scheduler_config.scheduler_training_steps / inference_steps_);
    delta_half_ = delta_ / 2;

    // leading spacing: t_i = round(i * delta)
    std::vector<int64_t> base_ts_(inference_steps_);
    for (uint64_t i = 0; i < inference_steps_; ++i) {
        base_ts_[i] = int64_t(std::llround(double(i) * double(delta_)));
    }

    // prk quarters, verbatim from diffusers:
    // prk = base_ts[-k:].repeat(2) + tile([0, delta/2], k); prk = (prk[:-1].repeat(2)[1:-1])[::-1]
    uint64_t k_ = std::min<uint64_t>(4, inference_steps_);
    std::vector<int64_t> prk_;
    {
        std::vector<int64_t> tail_(base_ts_.end() - k_, base_ts_.end());
        std::vector<int64_t> tiled_;
        for (uint64_t i = 0; i < k_; ++i) {
            tiled_.push_back(tail_[i]);
            tiled_.push_back(tail_[i] + delta_half_);
        }
        tiled_.pop_back();                                   // [:-1]
        std::vector<int64_t> doubled_;
        for (int64_t v_ : tiled_) { doubled_.push_back(v_); doubled_.push_back(v_); }  // repeat(2)
        prk_.assign(doubled_.begin() + 1, doubled_.end() - 1); // [1:-1]
        std::reverse(prk_.begin(), prk_.end());
    }
    prk_total_steps_ = long(prk_.size());                    // 4 * (k-1)
    for (long g_ = 0; g_ * 4 < prk_total_steps_; ++g_) {
        prk_anchor_ts_.push_back(prk_[size_t(g_ * 4)]);
    }

    // plms = base_ts[:-3] reversed
    std::vector<int64_t> plms_;
    {
        uint64_t keep_ = (inference_steps_ > 3) ? inference_steps_ - 3 : 0;
        for (uint64_t i = 0; i < keep_; ++i) plms_.push_back(base_ts_[i]);
        std::reverse(plms_.begin(), plms_.end());
    }

    // full internal sequence + matching sigmas for base scale/x0-conversion
    std::map<long, int64_t> expanded_timesteps_;
    vector<float> expanded_sigmas_;
    long idx_ = 0;
    for (int64_t t_ : prk_) {
        expanded_timesteps_.insert(make_pair(idx_, t_));
        expanded_sigmas_.push_back(generate_sigma_at(float(t_)));
        idx_++;
    }
    for (int64_t t_ : plms_) {
        expanded_timesteps_.insert(make_pair(idx_, t_));
        expanded_sigmas_.push_back(generate_sigma_at(float(t_)));
        idx_++;
    }
    scheduler_timesteps = expanded_timesteps_;
    scheduler_sigmas    = expanded_sigmas_;

    // init-noise alignment: leading spacing starts at t = (n-1)*delta < 999,
    // so the base σ_max (=σ(999)) no longer matches the first eval timestep.
    // diffusers feeds raw ε to the UNet at step 0 (PNDM does no input scaling);
    // framework UNet input is mask/√(σ_0²+1), hence set max_sigma = √(σ_0²+1)
    // so that scale(mask) == ε exactly at the first eval.
    float sigma_first_ = expanded_sigmas_[0];
    scheduler_max_sigma = std::sqrt(sigma_first_ * sigma_first_ + 1.0f);
    return uint64_t(idx_);
}

std::vector<float> PNDMDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    float random_intensity_
) {
    SD_UNUSED(random_intensity_);

    // recover genuine eps from base-converted x0: eps = (sample - x0) / σ_i
    float sigma_curs_ = scheduler_sigmas[size_t(step_index_)];
    PndmData curs_eps_(data_size_, 0.0f);
    for (long i = 0; i < data_size_; i++) {
        curs_eps_[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs_;
    }

    int64_t t_in_ = scheduler_timesteps[step_index_];

    if (pndm_counter_ < prk_total_steps_) {
        /* ---- Runge-Kutta quarter phase ---- */
        long diff_to_prev_ = (pndm_counter_ % 2 == 0) ? delta_half_ : 0;
        int64_t prev_t_ = t_in_ - diff_to_prev_;
        int64_t t_up_ = prk_anchor_ts_[size_t(pndm_counter_ / 4)];
        long phase_ = pndm_counter_ % 4;

        if (phase_ == 0) {
            cur_model_output_.assign(data_size_, 0.0f);
            for (long i = 0; i < data_size_; i++) cur_model_output_[i] = curs_eps_[i] / 6.0f;
            ets_.push_back(curs_eps_);
            if (ets_.size() > 4) ets_.erase(ets_.begin());
            cur_sample_.assign(samples_data_, samples_data_ + data_size_);
        } else if (phase_ == 1 || phase_ == 2) {
            for (long i = 0; i < data_size_; i++) cur_model_output_[i] += curs_eps_[i] / 3.0f;
        } else {
            for (long i = 0; i < data_size_; i++) curs_eps_[i] = cur_model_output_[i] + curs_eps_[i] / 6.0f;
            cur_model_output_.clear();
        }

        const float* anchor_ = cur_sample_.empty() ? samples_data_ : cur_sample_.data();
        float sigma_ref_  = generate_sigma_at(float(t_up_));
        float sigma_prev_ = generate_sigma_at(float(std::max<int64_t>(prev_t_, 0)));
        pndm_counter_++;
        return get_prev_sample(curs_eps_.data(), anchor_, data_size_, sigma_ref_, sigma_prev_);
    } else {
        /* ---- PLMS phase ---- */
        int64_t prev_t_ = t_in_ - delta_;
        ets_.push_back(curs_eps_);
        if (ets_.size() > 4) ets_.erase(ets_.begin());

        PndmData& e1_ = ets_.back();
        switch (ets_.size()) {
            case 1:
                break;                                       // e = e1
            case 2: {
                PndmData& e2_ = ets_[ets_.size() - 2];
                for (long i = 0; i < data_size_; i++) curs_eps_[i] = (3.0f * e1_[i] - e2_[i]) / 2.0f;
                break;
            }
            case 3: {
                PndmData& e2_ = ets_[ets_.size() - 2];
                PndmData& e3_ = ets_[ets_.size() - 3];
                for (long i = 0; i < data_size_; i++) curs_eps_[i] = (23.0f * e1_[i] - 16.0f * e2_[i] + 5.0f * e3_[i]) / 12.0f;
                break;
            }
            default: {
                PndmData& e2_ = ets_[ets_.size() - 2];
                PndmData& e3_ = ets_[ets_.size() - 3];
                PndmData& e4_ = ets_[ets_.size() - 4];
                for (long i = 0; i < data_size_; i++) curs_eps_[i] = (55.0f * e1_[i] - 59.0f * e2_[i] + 37.0f * e3_[i] - 9.0f * e4_[i]) / 24.0f;
                break;
            }
        }

        pndm_counter_++;
        float sigma_ref_  = generate_sigma_at(float(t_in_));
        // last PLMS step hits prev_t < 0: diffusers uses final_alpha_cumprod
        // (= alphas_cumprod[0] when set_alpha_to_one=false), i.e. σ(0)
        float sigma_prev_ = generate_sigma_at(float(std::max<int64_t>(prev_t_, 0)));
        return get_prev_sample(curs_eps_.data(), samples_data_, data_size_, sigma_ref_, sigma_prev_);
    }
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_PNDM
