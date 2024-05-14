/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_BASE_H
#define SCHEDULER_BASE_H

#include "onnxsd_defs.h"
#include "onnxsd_basic_core_config.cc"
#include "onnxsd_basic_register.cc"
#include "onnxsd_basic_tools.cc"

namespace onnx {
namespace sd {
namespace scheduler {

using namespace base;
using namespace amon;
using namespace Ort;
using namespace detail;

typedef enum BetaScheduleType {
    BETA_TYPE_LINEAR            = 1,
    BETA_TYPE_SCALED_LINEAR     = 2,
    BETA_TYPE_SQUAREDCOS_CAP_V2 = 3,
} BetaType;

typedef enum AlphaTransformType {
    ALPHA_TYPE_COSINE           = 1,
    ALPHA_TYPE_EXP              = 2,
} AlphaType;

typedef enum PredictionType {
    PREDICT_TYPE_EPSILON        = 1,
    PREDICT_TYPE_V_PREDICTION   = 2,
    PREDICT_TYPE_SAMPLE         = 3,
} PredictionType;

typedef struct SchedulerConfig {
    int scheduler_training_steps    = 1000;
    float scheduler_beta_start      = 0.00085f;
    float scheduler_beta_end        = 0.012f;
    uint64_t scheduler_seed         = 42;
    BetaType scheduler_beta_type    = BETA_TYPE_LINEAR;
    AlphaType scheduler_alpha_type  = ALPHA_TYPE_COSINE;
} SchedulerConfig;

class SchedulerBase {
private:
    RandomGenerator random_generator;

protected:
    SchedulerConfig scheduler_config{};
    PredictionType  scheduler_prediction_type;
    float           scheduler_max_sigma;
    vector<float>   scheduler_timesteps;
    vector<float>   scheduler_sigmas;
    vector<float>   alphas_cumprod;

protected:
    float generate_sigma_at(float timestep_);
    float generate_random_at(float timestep_);

    virtual std::vector<float> execute_method(
        const float* samples_data_, const float* predict_data_,
        int elements_in_batch, long step_index_, long order_) = 0;

public:
    explicit SchedulerBase(SchedulerConfig scheduler_config_ = {});
    virtual ~SchedulerBase();

    void create();
    void init(uint32_t inference_steps_) ;
    Tensor step(const Tensor& sample_, const Tensor& dnoise_, int timestep_, int order_ = 4);
    void release();
};

SchedulerBase::SchedulerBase(SchedulerConfig scheduler_config_){
    this->scheduler_max_sigma = 0;
    this->scheduler_config = scheduler_config_;
    this->scheduler_prediction_type = PredictionType::PREDICT_TYPE_EPSILON;
    this->random_generator.seed(scheduler_config_.scheduler_seed);
}

SchedulerBase::~SchedulerBase(){
    scheduler_max_sigma = 0;
    alphas_cumprod.clear();
    scheduler_sigmas.clear();
    scheduler_timesteps.clear();
    random_generator.~RandomGenerator();
}

float SchedulerBase::generate_random_at(float timestep_) {
    return random_generator.random_at(timestep_);
}

float SchedulerBase::generate_sigma_at(float timestep_) {
    int low_idx  = static_cast<int>(std::floor(timestep_));
    int high_idx = static_cast<int>(std::ceil(timestep_));
    float w      = timestep_ - static_cast<float>(low_idx);      // divide always 1
    float sigma  = (1.0f - w) * alphas_cumprod[low_idx] + w * alphas_cumprod[high_idx];
    return sigma;
}

void SchedulerBase::create() {
    int training_steps_  = scheduler_config.scheduler_training_steps;
    float linear_start_  = scheduler_config.scheduler_beta_start;
    float linear_end_    = scheduler_config.scheduler_beta_end;
    BetaType beta_type_  = scheduler_config.scheduler_beta_type;
    AlphaType alpha_type = scheduler_config.scheduler_alpha_type;

    switch (beta_type_) {
        case BETA_TYPE_LINEAR: {
            float beta_start_at = linear_start_;
            float beta_end_when = linear_end_;
            float beta_range = beta_end_when - beta_start_at;
            float product = 1.0f;

            for (uint32_t i = 0; i < training_steps_; ++i) {
                float beta_norm = beta_start_at + beta_range * ((float) i / float(training_steps_ - 1));
                product *= 1.0f - beta_norm;
                float comprod_sigma = std::powf((1 - product) / product, 0.5f);
                alphas_cumprod[i] = comprod_sigma;
            }
            break;
        }
        case BETA_TYPE_SCALED_LINEAR: {
            float beta_start_at = std::sqrtf(linear_start_);
            float beta_end_when = std::sqrtf(linear_end_);
            float beta_range = beta_end_when - beta_start_at;
            float product = 1.0f;

            for (uint32_t i = 0; i < training_steps_; ++i) {
                float beta_dire = beta_start_at + beta_range * ((float) i / float(training_steps_ - 1));
                float beta_norm = powf(beta_dire, 2.0f);
                product *= 1.0f - beta_norm;
                float comprod_sigma = std::powf((1 - product) / product, 0.5f);
                alphas_cumprod[i] = comprod_sigma;
            }
            break;
        }
        case BETA_TYPE_SQUAREDCOS_CAP_V2: {
            float beta_max = 0.999f;
            float product = 1.0f;

            auto alpha_bar_fn = [&](float f_step_) -> float {
                switch (alpha_type) {
                    case ALPHA_TYPE_COSINE:
                        return float(std::pow(std::cos((f_step_ + 0.008) / 1.008 * M_PI / 2), 2));
                    case ALPHA_TYPE_EXP:
                        return float(std::exp(f_step_ * -12.0f));
                    default:
                        return 1.0f;
                }
            };

            for (uint32_t i = 0; i < training_steps_; ++i) {
                float t1 = float(i) / float(training_steps_);
                float t2 = float(i + 1) / float(training_steps_);
                float beta_norm = std::min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), beta_max);
                product *= 1.0f - beta_norm;
                float comprod_sigma = std::powf((1 - product) / product, 0.5f);
                alphas_cumprod[i] = comprod_sigma;
            }
            break;
        }
        default: {
            render_report(class_exception(EXC_LOG_ERR, "ERROR:: PREDICT_TYPE_SAMPLE unimplemented"));
            break;
        }
    }
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
    int timestep_,
    int order_
) {
    // Get step index of timestep from TimeSteps
    long step_index_ = std::find(scheduler_timesteps.begin(), scheduler_timesteps.end(), timestep_) - scheduler_timesteps.begin();
    if (step_index_ == scheduler_timesteps.size()) {
        throw std::runtime_error("Timestep not found in TimeSteps.");
    }

    auto* sample_data_ = sample_.GetTensorData<float>();
    auto* dnoise_data_ = dnoise_.GetTensorData<float>();
    std::vector<int64_t> output_shape = sample_.GetTensorTypeAndShapeInfo().GetShape();
    size_t count = sample_.GetTensorTypeAndShapeInfo().GetElementCount();
    int elements_in_batch = (int)(output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]);
    float predict_data_[elements_in_batch];

    // do common prediction de-noise
    float sigma = scheduler_sigmas[step_index_];
    for (int i = 0; i < elements_in_batch; i++) {
        switch (scheduler_prediction_type) {
            case PREDICT_TYPE_EPSILON: {
                // predict_sample = sample - dnoise * sigma
                predict_data_[i] = sample_data_[i] - dnoise_data_[i] * sigma;
                break;
            }
            case PREDICT_TYPE_V_PREDICTION: {
                // c_out + input * c_skip
                predict_data_[i] = dnoise_data_[i] * (-sigma / std::sqrt(std::powf(sigma,2) + 1)) + (sample_data_[i] / (std::powf(sigma,2) + 1));
                break;
            }
            case PREDICT_TYPE_SAMPLE: {
                render_report(class_exception(EXC_LOG_ERR, "ERROR:: PREDICT_TYPE_SAMPLE unimplemented"));
                break;
            }
        }
    }

    std::vector<float> latent_data_ = execute_method(
        sample_data_, predict_data_, elements_in_batch, step_index_, order_
    );

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
