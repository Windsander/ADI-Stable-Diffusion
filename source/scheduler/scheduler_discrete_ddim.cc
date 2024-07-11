/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2024/07/09.
 *
 * Denoising Diffusion Implicit Models
 */
#ifndef SCHEDULER_DISCRETE_DDIM
#define SCHEDULER_DISCRETE_DDIM

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class DDIMDiscreteScheduler: public SchedulerBase {
private:
    RandomGenerator ddpm_random;

protected:
    std::vector<float> execute_method(
        const float* predict_data_,
        const float* samples_data_,
        long data_size_,
        long step_index_,
        float random_intensity_
    ) override;

public:
    explicit DDIMDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
        ddpm_random.seed(0);
    }

    ~DDIMDiscreteScheduler() override = default;
};

// from https://arxiv.org/pdf/2010.02502.pdf
// The engineering-enhanced version with partial parameter merging in accordance
// with the original formulas from the paper.
/**
 * x_{t-1} = sqrt(alpha_{t-1}) * ( (x_t - sqrt(1 - alpha_t) * epsilon_theta(x_t)) / sqrt(alpha_t) )
 *           + sqrt(1 - alpha_{t-1} - sigma_t^2) * epsilon_theta(x_t)
 *           + sigma_t * epsilon_t
 *
 * x_{t-1} = sqrt(alpha_{t-1}) * ( (x_t - sqrt(1 - alpha_t) * epsilon_theta(x_t)) / sqrt(alpha_t) )
 *            \_________________/  \_______________________________________________________________/
 *                "predicted x_0"                  "direction pointing to x_t"
 *
 *           + sqrt(1 - alpha_{t-1} - sigma_t^2) * epsilon_theta(x_t)
 *            \_________________________________/  \________________/
 *                "direction pointing to x_t"         "direction pointing to x_t"
 *
 *           + sigma_t * epsilon_t
 *            \________/  \_______/
 *            "sigma_t"  "random noise"
 */
std::vector<float> DDIMDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    float random_intensity_
) {
    std::vector<float> scaled_sample_(data_size_);

    // DDIM:: sigma get
    float sigma_curs = scheduler_sigmas[step_index_];
    float sigma_next = scheduler_sigmas[step_index_ + 1];
    float variance = 0;
    float revert_a = 0;
    float revert_b = 0;
    float factor_a = 0;
    float factor_b = 0;
    float eta = random_intensity_;      // DDIM use η=0, and when η=1, DDIM degrade to DDPM

    // combine calculated make wrong output below, only η=1 is available, by params.
    // The cancellation of hyperparameters during below computation process ensures
    // that DDIM with eta = 1 correctly reduces to DDPM during inference.
    // although it's the wrong params at first, so when η<1, the error occurs
    // { Why do the transformed hyperparameters of Sigma cause issues? Because DDPM relies
    //   on a Markov chain, whereas DDIM changes the Markov property that DDPM depends on to
    //   rely on the consistency of the Gaussian distribution through Gaussian distribution
    //   features. In our use of hyperparameters, we exceeded the original distribution mean,
    //   breaking the consistency and causing the noise to remain unresolved.}
    /* <Deprecated>
     * {
     *   float sigma_curs_pow = sigma_curs * sigma_curs;
     *   float sigma_next_pow = sigma_next * sigma_next;
     *   variance = (eta <= 0) ? 0 : eta * std::sqrt(sigma_next_pow / sigma_curs_pow * (sigma_curs_pow - sigma_next_pow) / (sigma_next_pow + 1));
     *   revert_a = std::sqrt(sigma_curs_pow + 1);
     *   factor_a = std::sqrt((sigma_curs_pow + 1) / (sigma_next_pow + 1));
     *   factor_b = std::sqrt(sigma_next_pow / (sigma_next_pow + 1) - variance * variance) -
     *              std::sqrt(sigma_curs_pow / (sigma_next_pow + 1));
     * }
     *
     * // DDIM:: current noise decrees
     * auto [c_skip, c_out, c_unused] = find_predict_params_at(sigma_curs);
     * for (int i = 0; i < data_size_; i++) {
     *   scaled_sample_[i] = (predict_data_[i] - samples_data_[i] * c_skip * revert_a) / c_out;     // get dnoised_data
     *   scaled_sample_[i] = samples_data_[i] * factor_a + scaled_sample_[i] * factor_b;
     *   if (sigma_next > 0 & eta > 0) { // η=1, DDIM should degrade to DDPM
     *       // so when η=1, factor_b = (sigma_next_pow - sigma_curs_pow) / (sigma_curs * std::sqrt(sigma_next_pow + 1));
     *       scaled_sample_[i] = scaled_sample_[i] + ddpm_random.next() * variance;
     *   }
     * }
     * </Deprecated>
     */
    {
        float sigma_curs_pow = sigma_curs * sigma_curs;
        float sigma_next_pow = sigma_next * sigma_next;
        variance = (eta <= 0) ? 0 : eta * std::sqrt(sigma_next_pow / sigma_curs_pow * (sigma_curs_pow - sigma_next_pow) / (sigma_next_pow + 1));
        factor_a = std::sqrt(sigma_next_pow + 1);
        factor_b = std::sqrt(sigma_next_pow / (sigma_next_pow + 1) - variance * variance) ;
    }

    // DDIM:: current noise decrees
    auto [c_skip, c_out, c_unused] = find_predict_params_at(sigma_curs);
    for (int i = 0; i < data_size_; i++) {
        scaled_sample_[i] = (predict_data_[i] - samples_data_[i] * c_skip) / c_out;      // get dnoised_data
        scaled_sample_[i] = predict_data_[i] * factor_a + scaled_sample_[i] * factor_b;
        if (sigma_next > 0 & eta > 0) { // η=1, DDIM should degrade to DDPM
            // so when η=1, factor_b = (sigma_next_pow - sigma_curs_pow) / (sigma_curs * std::sqrt(sigma_next_pow + 1));
            scaled_sample_[i] = scaled_sample_[i] + ddpm_random.next() * variance;
        }
    }

    return scaled_sample_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_DDIM
