/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_BASE_H
#define SCHEDULER_BASE_H

#include "onnxsd_foundation.cc"

namespace onnx {
namespace sd {
namespace scheduler {

using namespace base;
using namespace amon;

class SchedulerBase {
private:
    RandomGenerator random_generator;

protected:
    typedef std::tuple<float, float, float> Predictants;

protected:
    SchedulerConfig scheduler_config = DEFAULT_SCHEDULER_CONFIG;
    std::map<long, int64_t> scheduler_timesteps;
    vector<float> scheduler_sigmas;
    vector<float> alphas_cumprod;
    float scheduler_max_sigma;

protected:
    Predictants find_predict_params_at(float sigma_) ;
    long find_closest_timestep_index(long time_);
    float generate_sigma_at(float timestep_);

protected:
    virtual uint64_t correction_steps(uint64_t inference_steps_) { return inference_steps_; };
    virtual std::vector<float> execute_method(
        const float *predict_data_, const float* samples_data_,
        long data_size_, long step_index_, float random_intensity_) = 0;

public:
    explicit SchedulerBase(const SchedulerConfig &scheduler_config_ = DEFAULT_SCHEDULER_CONFIG);
    virtual ~SchedulerBase();

    void create();
    uint64_t init(uint64_t inference_steps_) ;
    Tensor mask(const TensorShape& mask_shape_);
    Tensor scale(const Tensor& masker_, int step_index_);
    Tensor time(int step_index_);
    Tensor step(const Tensor& sample_, const Tensor& dnoise_, int step_index_, float random_intensity_ = 1.0f);
    void uninit();
    void release();
};

SchedulerBase::SchedulerBase(const SchedulerConfig& scheduler_config_){
    this->scheduler_max_sigma = 0;
    this->scheduler_config = scheduler_config_;
    this->random_generator.seed(scheduler_config_.scheduler_seed);
}

SchedulerBase::~SchedulerBase(){
    scheduler_max_sigma = 0;
    alphas_cumprod.clear();
    scheduler_sigmas.clear();
    scheduler_timesteps.clear();
    random_generator.~RandomGenerator();
}

long SchedulerBase::find_closest_timestep_index(long time_) {
    auto it = scheduler_timesteps.lower_bound(time_);
    if (it == scheduler_timesteps.end()) {
        throw std::runtime_error("closest index found failed");
    }
    if ((it != scheduler_timesteps.begin()) &&
        (it == scheduler_timesteps.end() ||
         std::abs(it->second - time_) >= std::abs(std::prev(it)->second - time_))) {
        --it;
    }
    return it->first;
}

float SchedulerBase::generate_sigma_at(float timestep_) {
    int low_idx   = static_cast<int>(std::floor(timestep_));
    int high_idx  = static_cast<int>(std::ceil(timestep_));
    float l_sigma = alphas_cumprod[low_idx];
    float h_sigma = alphas_cumprod[high_idx];
    float w       = timestep_ - static_cast<float>(low_idx);      // divide always 1
    float alpha_prod = (1.0f - w) * l_sigma + w * h_sigma;
    float sigma = std::pow((1 - alpha_prod) / alpha_prod, 0.5f);
    // Mark: for safety & efficiency, I'm using [our_sigma^2 = sigma^2/(1-sigma^2)]
    return sigma;
}

SchedulerBase::Predictants SchedulerBase::find_predict_params_at(float sigma_)
{
    float c_skip, c_out;
    {
        switch (scheduler_config.scheduler_predict_type) {
            case PREDICT_TYPE_EPSILON: {
                // predict_sample = sample - dnoise * sigma
                c_skip = 1.0f;
                c_out = -sigma_;
                break;
            }
            case PREDICT_TYPE_V_PREDICTION: {
                // predict_sample = sample * alpha_prod^2 - c_out * beta_prod^2
                c_skip = float(1.0f / (std::pow(sigma_, 2) + 1));
                c_out = -float(sigma_ / std::sqrt(std::pow(sigma_, 2) + 1));
                break;
            }
            case PREDICT_TYPE_SAMPLE: {
                // predict_sample = dnoise
                c_skip = 0.0f;
                c_out = +1.0f;
                break;
            }
            default: {
                amon_report(class_exception(EXC_LOG_ERR, "ERROR:: Unknown prediction type"));
                return {};
            }
        }
    }
    return std::make_tuple(c_skip, c_out, 0.0f);
}

void SchedulerBase::create() {
    uint64_t training_steps_  = scheduler_config.scheduler_training_steps;
    float linear_start_  = scheduler_config.scheduler_beta_start;
    float linear_end_    = scheduler_config.scheduler_beta_end;
    BetaType beta_type_  = scheduler_config.scheduler_beta_type;
    AlphaType alpha_type = scheduler_config.scheduler_alpha_type;

    switch (beta_type_) {
        case BETA_TYPE_LINEAR: {
            float beta_start_at = linear_start_;
            float beta_end_when = linear_end_;
            float beta_range = beta_end_when - beta_start_at;
            float alpha_prod = 1.0f;

            for (uint32_t i = 0; i < training_steps_; ++i) {
                float beta_norm = beta_start_at + beta_range * ((float) i / float(training_steps_ - 1));
                alpha_prod *= 1.0f - beta_norm;
                alphas_cumprod.push_back(alpha_prod);
            }
            break;
        }
        case BETA_TYPE_SCALED_LINEAR: {
            float beta_start_at = std::sqrt(linear_start_);
            float beta_end_when = std::sqrt(linear_end_);
            float beta_range = beta_end_when - beta_start_at;
            float alpha_prod = 1.0f;

            for (uint32_t i = 0; i < training_steps_; ++i) {
                float beta_dire = beta_start_at + beta_range * ((float) i / float(training_steps_ - 1));
                float beta_norm = pow(beta_dire, 2.0f);
                alpha_prod *= 1.0f - beta_norm;
                alphas_cumprod.push_back(alpha_prod);
            }
            break;
        }
        case BETA_TYPE_SQUAREDCOS_CAP_V2: {
            float beta_max = 0.999f;
            float alpha_prod = 1.0f;

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
                float beta_norm = min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), beta_max);
                alpha_prod *= 1.0f - beta_norm;
                alphas_cumprod.push_back(alpha_prod);
            }
            break;
        }
        default: {
            amon_report(class_exception(EXC_LOG_ERR, "ERROR:: PREDICT_TYPE_SAMPLE unimplemented"));
            break;
        }
    }
}

uint64_t SchedulerBase::init(uint64_t inference_steps_) {
    std::vector<float> result;
    if (inference_steps_ == 0) {
        amon_report(class_exception(EXC_LOG_ERR, "ERROR:: inference_steps_ setting with 0!"));
        return 0;
    }

    // linearspace
    int start_at = 0;
    int end_when = int(scheduler_config.scheduler_training_steps - 1);
    float step_gap = (inference_steps_ > 1) ?
                     float(end_when - start_at) / float(inference_steps_ - 1) :
                     float(end_when);

    for (uint32_t i = 0; i < inference_steps_; ++i) {
        float t = float(end_when) - step_gap * float(i);
        float sigma = generate_sigma_at(t);
        scheduler_timesteps.insert(make_pair(long(i), int64_t(t)));
        scheduler_sigmas.push_back(sigma);
        scheduler_max_sigma = max(scheduler_max_sigma, sigma);
    }
    scheduler_sigmas.push_back(0);
    return correction_steps(inference_steps_);
}

Tensor SchedulerBase::mask(const TensorShape& mask_shape_){
    return TensorHelper::random<float>(mask_shape_, random_generator, scheduler_max_sigma);
}

Tensor SchedulerBase::scale(const Tensor& latent_, int step_index_){
    // Get step index of timestep from TimeSteps
    if (step_index_ >= scheduler_timesteps.size()) {
        throw std::runtime_error("from time not found target TimeSteps.");
    }
    float sigma = scheduler_sigmas[step_index_];
    sigma = std::sqrt(sigma * sigma + 1);
    return TensorHelper::divide<float>(latent_, sigma);
}

Tensor SchedulerBase::time(int step_index_){
    // Get step index of timestep from TimeSteps
    if (step_index_ >= scheduler_timesteps.size()) {
        throw std::runtime_error("from time not found target TimeSteps.");
    }
    vector<int64_t> timestep_value_{scheduler_timesteps[step_index_]};
    TensorShape timestep_shape_{1};
    return TensorHelper::create<int64_t>(timestep_shape_, timestep_value_);
}

Tensor SchedulerBase::step(
    const Tensor& sample_,
    const Tensor& dnoise_,
    int step_index_,
    float random_intensity_
) {
    // Check step index of timestep from TimeSteps
    if (step_index_ >= scheduler_timesteps.size()) {
        throw std::runtime_error("from time not found target TimeSteps.");
    }

    TensorShape output_shape_ = sample_.GetTensorTypeAndShapeInfo().GetShape();
    long data_size_ = TensorHelper::get_data_size(sample_);
    auto* sample_data_ = sample_.GetTensorData<float>();
    auto* dnoise_data_ = dnoise_.GetTensorData<float>();
    std::vector<float> predict_data_(data_size_);

    // do common prediction de-noise
    float sigma = scheduler_sigmas[step_index_];
    auto [c_skip, c_out, c_unused] = find_predict_params_at(sigma);
    for (int i = 0; i < data_size_; i++) {
        // predict_sample = sample * c_skip + c_out * dnoise
        predict_data_[i] = sample_data_[i] * c_skip + dnoise_data_[i] * c_out;
    }

    std::vector<float> latent_value_ = execute_method(
        predict_data_.data(), sample_data_, data_size_, step_index_, random_intensity_
    );
    Tensor result_latent = TensorHelper::create(output_shape_, latent_value_);

    return result_latent;
}

void SchedulerBase::uninit() {
    scheduler_timesteps.clear();
    scheduler_sigmas.clear();
}

void SchedulerBase::release() {
    alphas_cumprod.clear();
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_BASE_H
