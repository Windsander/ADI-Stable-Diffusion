/*
 * Copyright (c) 2018-2050 SD_UNet - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_UNET_H
#define MODEL_UNET_H

#include "model_base.cc"

namespace onnx {
namespace sd {
namespace units {

using namespace base;
using namespace amon;
using namespace scheduler;
using namespace Ort;
using namespace detail;

#define DEFAULT_UNET_CONDIG                                          \
    {                                                                \
        /*sd_scheduler_config*/ DEFAULT_SCHEDULER_CONDIG,            \
        /*sd_scheduler_type*/   SchedulerType::SCHEDULER_EULAR_A,    \
        /*sd_inference_steps*/  3,                                   \
        /*sd_input_width*/      512,                                 \
        /*sd_input_height*/     512,                                 \
        /*sd_input_channel*/    4,                                   \
        /*sd_scale_guidance*/   0.9f,                                \
        /*sd_scale_positive*/   0.9f                                 \
    }                                                                \

typedef struct ModelUNetConfig {
    SchedulerConfig sd_scheduler_config;
    SchedulerType sd_scheduler_type;
    uint64_t sd_inference_steps;
    uint64_t sd_input_width;
    uint64_t sd_input_height;
    uint64_t sd_input_channel;
    float sd_scale_guidance;
    float sd_scale_positive;
} ModelUNetConfig;

class UNet : public ModelBase {
private:
    ModelUNetConfig sd_unet_config = DEFAULT_UNET_CONDIG;
    SchedulerEntity_ptr sd_scheduler_p;

public:
    explicit UNet(std::string model_path_, const ModelUNetConfig &unet_config_ = DEFAULT_UNET_CONDIG);
    ~UNet() override;

    Tensor inference(const Tensor &emb_positive_,const Tensor &emb_negative_, const Tensor &encoded_img_);
};

UNet::UNet(std::string model_path_, const ModelUNetConfig& unet_config_) : ModelBase(std::move(model_path_)){
    sd_unet_config = unet_config_;
    sd_scheduler_p = SchedulerRegister::request_scheduler(
        unet_config_.sd_scheduler_type,
        unet_config_.sd_scheduler_config
    );
    sd_scheduler_p->init(unet_config_.sd_inference_steps);
}

UNet::~UNet(){
    sd_scheduler_p->uninit();
    sd_scheduler_p = SchedulerRegister::recycle_scheduler(sd_scheduler_p);
}

Tensor UNet::inference(
    const Tensor &emb_positive_,
    const Tensor &emb_negative_,
    const Tensor &encoded_img_
) {

    int w_ = int(sd_unet_config.sd_input_width);
    int h_ = int(sd_unet_config.sd_input_height);
    int c_ = int(sd_unet_config.sd_input_channel);

    TensorShape latent_shape_{1, c_, h_ / 8, w_ / 8};
    std::vector<float> latent_empty_{};
    Tensor latents_ =  (encoded_img_.HasValue())?
        TensorHelper::duplicate(encoded_img_, latent_shape_):
        TensorHelper::create(latent_shape_, latent_empty_);
    Tensor init_mask_ = sd_scheduler_p->mask(latent_shape_);

    for (int i = 0; i < sd_unet_config.sd_inference_steps; ++i) {
        Tensor model_latent_ =  (latents_.HasValue())?
            TensorHelper::add(latents_, sd_scheduler_p->scale(init_mask_, i), latent_shape_) :
            sd_scheduler_p->scale(init_mask_, i);
        Tensor timestep_ = sd_scheduler_p->time(i);

        // do positive
        Tensor guided_pred_positive_ =TensorHelper::create(latent_shape_, latent_empty_);
        if (emb_positive_.HasValue()){
            std::vector<Tensor> input_tensors;
            Ort::AllocatorWithDefaultOptions ort_alloc;
            input_tensors.push_back(emb_positive_.GetValue(0, ort_alloc));
            input_tensors.push_back(model_latent_.GetValue(0, ort_alloc));
            input_tensors.push_back(timestep_.GetValue(0, ort_alloc));
            std::vector<Tensor> output_tensors;
            execute(input_tensors, output_tensors);

            // Split output_tensors from [2, 4, 64, 64] to [1, 4, 64, 64]
            std::vector<Tensor> model_result_positive_;
            model_result_positive_ = TensorHelper::split(output_tensors[0].GetValue(0, ort_alloc));
            guided_pred_positive_ = TensorHelper::guidance(
                model_result_positive_[0], model_result_positive_[1], sd_unet_config.sd_scale_guidance
            );
        }

        // do negative
        Tensor guided_pred_negative_ =TensorHelper::create(latent_shape_, latent_empty_);
        if (emb_negative_.HasValue()){
            std::vector<Tensor> input_tensors;
            Ort::AllocatorWithDefaultOptions ort_alloc;
            input_tensors.push_back(emb_negative_.GetValue(0, ort_alloc));
            input_tensors.push_back(model_latent_.GetValue(0, ort_alloc));
            input_tensors.push_back(timestep_.GetValue(0, ort_alloc));
            std::vector<Tensor> output_tensors;
            execute(input_tensors, output_tensors);

            // Split output_tensors from [2, 4, 64, 64] to [1, 4, 64, 64]
            std::vector<Tensor> model_result_negative_;
            model_result_negative_ = TensorHelper::split(output_tensors[0].GetValue(0, ort_alloc));
            guided_pred_negative_ = TensorHelper::guidance(
                model_result_negative_[0], model_result_negative_[1], sd_unet_config.sd_scale_guidance
            );
        }

        // do merge
        float merge_factor_ = sd_unet_config.sd_scale_positive;
        Tensor guided_pred_ = (
            (guided_pred_negative_.HasValue()) ?
            TensorHelper::guidance(guided_pred_negative_, guided_pred_positive_, merge_factor_) :
            TensorHelper::duplicate(guided_pred_positive_, latent_shape_)
        );

        latents_ = sd_scheduler_p->step(model_latent_, guided_pred_, i);
    }

    return latents_;
}


} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_UNET_H

