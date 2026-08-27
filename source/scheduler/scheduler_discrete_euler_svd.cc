/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2026/08/26.
 *
 * Euler discrete scheduler in its SVD (Stable Video Diffusion) configuration,
 * matching diffusers EulerDiscreteScheduler with svd-xt-1-1 scheduler_config:
 *   prediction_type = v_prediction, use_karras_sigmas = true,
 *   sigma_min = 0.002, sigma_max = 700.0, timestep_type = continuous
 *
 * Differences from the base EulerDiscrete path (all verified against
 * diffusers 0.39 objects, see GOAL_SESSION_v2.0.0 notes):
 *   - sigmas: karras rho=7 ramp between the EXPLICIT [sigma_max .. sigma_min]
 *     bounds (not derived from the beta schedule)
 *   - timesteps: continuous t = 0.25 * ln(sigma), emitted as float tensors
 *     (base class emits discrete int64 timesteps)
 *   - init noise: randn * sqrt(sigma_max^2 + 1) (= scheduler.init_noise_sigma)
 * Step / scale math (v_prediction x0, euler derivative) is inherited unchanged:
 *   x0    = sample / (sigma^2+1) - sigma/sqrt(sigma^2+1) * model_out
 *   next  = sample + (sample - x0)/sigma * (sigma_next - sigma)
 */
#ifndef SCHEDULER_DISCRETE_EULER_SVD
#define SCHEDULER_DISCRETE_EULER_SVD

#include "scheduler_discrete_euler.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class EulerSVDScheduler : public EulerDiscreteScheduler {
private:
    std::vector<float> scheduler_timesteps_f;   // continuous t = 0.25 * ln(sigma)

public:
    explicit EulerSVDScheduler(SchedulerConfig scheduler_config_ = {}) : EulerDiscreteScheduler(scheduler_config_){
        // SVD predicts v; keep the invariant inside the class so a misconfigured
        // CLI (--predictor epsilon) cannot silently corrupt the trajectory
        scheduler_config.scheduler_predict_type = PREDICT_TYPE_V_PREDICTION;
    }

    ~EulerSVDScheduler() override = default;

    uint64_t init(uint64_t inference_steps_) override {
        if (inference_steps_ == 0) {
            amon_report(class_exception(EXC_LOG_ERR, "ERROR:: inference_steps_ setting with 0!"));
            return 0;
        }

        float sigma_min_ = (scheduler_config.scheduler_sigma_min > 0) ?
                           scheduler_config.scheduler_sigma_min : 0.002f;
        float sigma_max_ = (scheduler_config.scheduler_sigma_max > 0) ?
                           scheduler_config.scheduler_sigma_max : 700.0f;

        // Karras et al. 2022 rho-schedule (rho=7) over explicit bounds,
        // identical to diffusers _convert_to_karras with config sigma_min/max
        const float karras_rho_ = 7.0f;
        double ramp_low_  = std::pow(double(sigma_max_), 1.0 / karras_rho_);
        double ramp_high_ = std::pow(double(sigma_min_), 1.0 / karras_rho_);
        for (uint32_t i = 0; i < inference_steps_; ++i) {
            double w = (inference_steps_ > 1) ? double(i) / double(inference_steps_ - 1) : 0.0;
            float sigma = float(std::pow(ramp_low_ + w * (ramp_high_ - ramp_low_), karras_rho_));
            scheduler_sigmas.push_back(sigma);
            scheduler_timesteps.insert(make_pair(long(i), int64_t(i)));   // index bookkeeping only
            scheduler_timesteps_f.push_back(0.25f * std::log(sigma));     // continuous timestep
        }
        scheduler_sigmas.push_back(0);
        // diffusers init_noise_sigma = (sigmas.max()^2 + 1)^0.5; base mask()
        // scales randn by scheduler_max_sigma, so store that exact value
        scheduler_max_sigma = float(std::sqrt(double(sigma_max_) * double(sigma_max_) + 1.0));
        return inference_steps_;
    }

    Tensor time(int step_index_) override {
        if (step_index_ >= scheduler_timesteps_f.size()) {
            throw std::runtime_error("from time not found target TimeSteps.");
        }
        std::vector<float> timestep_value_{scheduler_timesteps_f[step_index_]};
        TensorShape timestep_shape_{1};
        return TensorHelper::create<float>(timestep_shape_, timestep_value_);
    }

    void uninit() override {
        scheduler_timesteps_f.clear();
        SchedulerBase::uninit();
    }
};

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_EULER_SVD
