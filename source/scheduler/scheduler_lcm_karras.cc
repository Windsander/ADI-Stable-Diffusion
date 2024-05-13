/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_EULER_A_DISCRETE
#define SCHEDULER_EULER_A_DISCRETE

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class LCMKarrasScheduler: public SchedulerBase {
private:
    std::normal_distribution<float> random_style;

protected:
    float generate_random_at(float timestep) override;
    float generate_sigma_at(float timestep) override;

    std::vector<float> execute_method(
        const vector<float>& dnoised_data_,
        long step_index_
    ) override;

public:
    explicit LCMKarrasScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_){
        random_style =std::normal_distribution<float>(0.0, 1.0);
    }

    ~LCMKarrasScheduler() override{
        random_style.reset();
    }
};

float LCMKarrasScheduler::generate_random_at(float timestep_) {
    SD_UNUSED(timestep_);
    return random_style(random_generator);
}

float LCMKarrasScheduler::generate_sigma_at(float timestep_) {
    // TODO

    return 1.0f;
}

std::vector<float> LCMKarrasScheduler::execute_method(
    const vector<float>& dnoised_data_,
    long step_index_
) {
    std::vector<float> scaled_sample(dnoised_data_.size());

    // TODO

    return scaled_sample;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_EULER_A_DISCRETE
