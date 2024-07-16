/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2024/07/16.
 *
 * Unified Predictor-Corrector Method
 */
#ifndef SCHEDULER_DISCRETE_UNIPC
#define SCHEDULER_DISCRETE_UNIPC

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class UniPCDiscreteScheduler: public SchedulerBase {
private:
    typedef std::vector<float> UniData;
    std::vector<UniData> history_dnoise;
    UniData last_samples_;

private:
    //float get_unipc_beta_snr(float sigma_);
    long get_unified_history_count(long step_index_);
    UniData get_unified_correction(UniData curs_samples_, UniData curs_dnoised_, long prev_index_);
    UniData get_unified_prediction(UniData cors_samples_, UniData curs_dnoised_, long curs_index_);

protected:
    std::vector<float> execute_method(
        const float* predict_data_,
        const float* samples_data_,
        long data_size_,
        long step_index_,
        float random_intensity_
    ) override;

public:
    explicit UniPCDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
    }

    ~UniPCDiscreteScheduler() override = default;
};

/* Assistant Operations ===================================================*/

long UniPCDiscreteScheduler::get_unified_history_count(long step_index_){
    long maintain_order_ = long(scheduler_config.scheduler_maintain_cache);
    return std::min(maintain_order_, step_index_);
}

UniPCDiscreteScheduler::UniData UniPCDiscreteScheduler::get_unified_correction(
    UniData curs_samples_, UniData curs_dnoised_, long prev_index_
){
    long last_order_ = get_unified_history_count(prev_index_);
}

UniPCDiscreteScheduler::UniData UniPCDiscreteScheduler::get_unified_prediction(
    UniData cors_samples_, UniData curs_dnoised_, long curs_index_
){
    long curs_order_ = get_unified_history_count(curs_index_);
}

/* Essential Operations ===================================================*/
/**
 * base on: https://arxiv.org/pdf/2302.04867
 */
std::vector<float> UniPCDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    float random_intensity_
) {
    SD_UNUSED(random_intensity_);

    std::vector<float> next_samples_(data_size_);
    std::vector<float> curs_dnoised_(data_size_);
    std::vector<float> curs_samples_(data_size_);
    long maintain_order_ = long(scheduler_config.scheduler_maintain_cache);

    // UniPC:: sigma get
    float sigma_curs = scheduler_sigmas[step_index_];

    // UniPC: get current model output as M_t
    auto [c_skip, c_out, c_unused] = find_predict_params_at(sigma_curs);
    for (int i = 0; i < data_size_; i++) {
        curs_dnoised_[i] = (predict_data_[i] - samples_data_[i] * c_skip) / c_out;
        curs_samples_[i] = samples_data_[i];
    }

    // UniPC: do unified correction logic
    curs_samples_ = get_unified_correction(curs_samples_, curs_dnoised_, step_index_ - 1);

    // UniPC: update history records, insert M_t to records->m[0]
    {
        history_dnoise.insert(history_dnoise.begin(), curs_dnoised_);
        if (history_dnoise.size() > maintain_order_) {
            history_dnoise.pop_back();
        }
        last_samples_ = curs_samples_;
    }

    // UniPC: do unified prediction logic
    next_samples_ = get_unified_prediction(curs_samples_, curs_dnoised_, step_index_);

    return next_samples_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_UNIPC
