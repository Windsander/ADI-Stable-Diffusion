/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_BASE_H
#define SCHEDULER_BASE_H

#include "onnxsd_defs.h"
#include "onnxsd_basic_register.cc"

namespace onnx {
namespace sd {
namespace scheduler {

using namespace base;
using namespace amon;
using namespace Ort;
using namespace detail;

typedef enum PredictionType {
    PREDICT_TYPE_EPSILON        = 1,
    PREDICT_TYPE_V_PREDICTION   = 2,
    PREDICT_TYPE_SAMPLE         = 3,
} PredictionType;

typedef struct SchedulerConfig {
    int scheduler_training_steps = 1000;
    float scheduler_beta_start   = 0.00085f;
    float scheduler_beta_end     = 0.012f;
    uint64_t scheduler_seed      = 42;
    bool scheduler_beta_scale    = false;
} SchedulerConfig;

class SchedulerBase {
protected:
    std::default_random_engine random_generator;

protected:
    SchedulerConfig scheduler_config{};
    PredictionType  scheduler_prediction_type;
    float           scheduler_max_sigma;
    vector<float>   scheduler_timesteps;
    vector<float>   scheduler_sigmas;
    vector<float>   alphas_cumprod;

protected:
    float sigma_to_timestep(float sigma);

    virtual float generate_random_at(float timestep) = 0;
    virtual float generate_sigma_at(float timestep) = 0;
    virtual std::vector<float> execute_method(const vector<float>& dnoised_data_, long step_index_) = 0;

public:
    explicit SchedulerBase(SchedulerConfig scheduler_config_ = {});
    virtual ~SchedulerBase();

    void create(int training_steps, float linear_start_, float linear_end_, bool enable_scale_ = false);
    void init(uint32_t inference_steps_) ;
    Tensor step(const Tensor& sample_, const Tensor& dnoise_, int timestep);
    void release();
};

SchedulerBase::SchedulerBase(SchedulerConfig scheduler_config_){
    this->scheduler_max_sigma = 0;
    this->scheduler_config = scheduler_config_;
    this->scheduler_prediction_type = PredictionType::PREDICT_TYPE_EPSILON;
    this->random_generator.seed(scheduler_config_.scheduler_seed);
    create(
        scheduler_config_.scheduler_training_steps,
        scheduler_config_.scheduler_beta_start,
        scheduler_config_.scheduler_beta_end,
        scheduler_config_.scheduler_beta_scale
    );
}

SchedulerBase::~SchedulerBase(){
    scheduler_max_sigma = 0;
    alphas_cumprod.clear();
    scheduler_sigmas.clear();
    scheduler_timesteps.clear();
}

void SchedulerBase::create(
    int training_steps, float linear_start_, float linear_end_, bool enable_scale_
) {
    float beta_start_at = enable_scale_? std::sqrtf(linear_start_):linear_start_;
    float beta_end_when = enable_scale_? std::sqrtf(linear_end_):linear_end_;
    float beta_range = beta_end_when - beta_start_at;
    float product = 1.0f;

    for (uint32_t i = 0; i < training_steps; ++i) {
        float beta_dire = beta_start_at + beta_range * ((float) i / float(training_steps - 1));
        float beta_norm = enable_scale_ ? powf(beta_dire, 2.0f) : beta_dire;
        product *= 1.0f - beta_norm;
        float comprod_sigma = std::powf((1 - product) / product, 0.5f);
        alphas_cumprod[i] = comprod_sigma;
    }
}

float SchedulerBase::sigma_to_timestep(float sigma){
    float log_sigma = std::log(sigma);
    std::vector<float> dists;
    dists.reserve(scheduler_config.scheduler_training_steps);
    for (float log_sigma_val : scheduler_sigmas) {
        dists.push_back(log_sigma - log_sigma_val);
    }

    int training_steps = scheduler_config.scheduler_training_steps;
    int low_idx = 0;
    for (size_t i = 0; i < training_steps; i++) {
        if (dists[i] >= 0) {
            low_idx++;
        }
    }
    low_idx      = std::min(std::max(low_idx - 1, 0), training_steps - 2);
    int high_idx = low_idx + 1;

    float low  = scheduler_sigmas[low_idx];
    float high = scheduler_sigmas[high_idx];
    float w    = std::max(0.f, std::min(1.f, (low - log_sigma) / (low - high)));
    int t    = int((1.0f - w) * low_idx + w * high_idx);

    return t;
}

void SchedulerBase::init(uint32_t inference_steps_) {
    std::vector<float> result;
    if (inference_steps_ == 0) {
        render_report(class_exception(EXC_LOG_ERR, "ERROR:: inference_steps_ setting with 0!"));
        return;
    }

    // linearspace
    int start_at = 0;
    int end_when = scheduler_config.scheduler_training_steps - 1;
    float step_gap = (inference_steps_ > 1) ?
                     float(end_when - start_at) / float(inference_steps_ - 1) :
                     float(end_when);

    for (uint32_t i = 0; i < inference_steps_; ++i) {
        float t = float(end_when) - step_gap * float(i);
        float sigma = generate_sigma_at(t);
        scheduler_timesteps.push_back(t);
        scheduler_sigmas.push_back(sigma);
        scheduler_max_sigma = std::max(scheduler_max_sigma, sigma);
    }
    scheduler_sigmas.push_back(0);
}

Tensor SchedulerBase::step(
    const Tensor& sample_,
    const Tensor& dnoise_,
    int timestep_
) {
    // Get step index of timestep from TimeSteps
    long step_index_ = std::find(scheduler_timesteps.begin(), scheduler_timesteps.end(), timestep_) - scheduler_timesteps.begin();
    if (step_index_ == scheduler_timesteps.size()) {
        throw std::runtime_error("Timestep not found in TimeSteps.");
    }

    auto* sample_data = sample_.GetTensorData<float>();
    auto* dnoise_data = dnoise_.GetTensorData<float>();
    std::vector<int64_t> output_shape = sample_.GetTensorTypeAndShapeInfo().GetShape();
    size_t count = sample_.GetTensorTypeAndShapeInfo().GetElementCount();
    int elements_in_batch = (int)(output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]);
    std::vector<float> dnoised_data_(elements_in_batch);

    // do common prediction de-noise
    float sigma = scheduler_sigmas[step_index_];
    for (int i = 0; i < elements_in_batch; i++) {
        switch (scheduler_prediction_type) {
            case PREDICT_TYPE_EPSILON: {
                // predict_sample = sample - dnoise * sigma
                dnoised_data_[i] = (sample_data[i] - dnoise_data[i] * sigma);
                break;
            }
            case PREDICT_TYPE_V_PREDICTION: {
                // c_out + input * c_skip
                dnoised_data_[i] =  dnoise_data[i] * (-sigma / std::sqrt(std::pow(sigma,2) + 1)) + (sample_data[i] / (std::pow(sigma,2) + 1));
                break;
            }
            case PREDICT_TYPE_SAMPLE: {
                render_report(class_exception(EXC_LOG_ERR, "ERROR:: PREDICT_TYPE_SAMPLE unimplemented"));
                break;
            }
        }
    }

    std::vector<float> latent_data_ = execute_method(dnoised_data_, step_index_);

    Tensor result_latent = Value::CreateTensor(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault).GetConst(),
        &latent_data_[0], latent_data_.size(),
        &output_shape[0], count
    );

    return result_latent;
}

void SchedulerBase::release() {
    alphas_cumprod.clear();
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_BASE_H
