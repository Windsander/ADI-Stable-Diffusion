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
        /*sd_scale_guidance*/   0.9f                                 \
    }                                                                \

typedef struct ModelUNetConfig {
    SchedulerConfig sd_scheduler_config;
    SchedulerType sd_scheduler_type;
    uint64_t sd_inference_steps;
    uint64_t sd_input_width;
    uint64_t sd_input_height;
    uint64_t sd_input_channel;
    float sd_scale_guidance;
} ModelUNetConfig;

class UNet : public ModelBase {
private:
    ModelUNetConfig sd_unet_config;
    SchedulerEntity_ptr sd_scheduler_p;

public:
    explicit UNet(std::string model_path_, const ModelUNetConfig &unet_config_ = DEFAULT_UNET_CONDIG);
    ~UNet() override;

    Tensor inference(const Tensor &emb_positive_,const Tensor &emb_negetive_, const Tensor &encoded_img_);
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
    const Tensor &emb_negetive_,
    const Tensor &encoded_img_
) {

    int w_ = int(sd_unet_config.sd_input_width);
    int h_ = int(sd_unet_config.sd_input_height);
    int c_ = int(sd_unet_config.sd_input_channel);

    TensorShape latent_shape_{1, c_, h_ / 8, w_ / 8};
    std::vector<float> latent_empty_{};
    Tensor latents_ = TensorHelper::create(latent_shape_, latent_empty_);
    Tensor init_mask_ = sd_scheduler_p->mask(latent_shape_);

    for (int i = 0; i < sd_unet_config.sd_inference_steps; ++i) {
        Tensor model_latent_ =  (encoded_img_.HasValue())?
            TensorHelper::add(encoded_img_, sd_scheduler_p->scale(init_mask_, i), latent_shape_) :
            sd_scheduler_p->scale(latents_, i);
        Tensor timestep_ = sd_scheduler_p->time(i);

        // do positive first
        Tensor guided_pred_positive_;
        {
            std::vector<Tensor> input_tensors;
            Ort::AllocatorWithDefaultOptions ort_alloc;
            input_tensors.push_back(emb_positive_.GetValue(0, ort_alloc));
            input_tensors.push_back(model_latent_.GetValue(0, ort_alloc));
            input_tensors.push_back(timestep_.GetValue(0, ort_alloc));
            std::vector<Tensor> output_tensors;
            execute(input_tensors, output_tensors);

            std::vector<Tensor> model_result_ ;
            // Split output_tensors from [2, 4, 64, 64] to [1, 4, 64, 64]
            Tensor &noise_pred_img_ = model_result_[0];
            Tensor &noise_pred_txt_ = model_result_[1];
            Tensor guided_pred_ = TensorHelper::guidance(
                noise_pred_img_, noise_pred_txt_, sd_unet_config.sd_scale_guidance
            );
        }

        Tensor guided_pred_negative_ ;
        {
            std::vector<Tensor> input_tensors;
            Ort::AllocatorWithDefaultOptions ort_alloc;
            input_tensors.push_back(emb_positive_.GetValue(0, ort_alloc));
            input_tensors.push_back(model_latent_.GetValue(0, ort_alloc));
            input_tensors.push_back(timestep_.GetValue(0, ort_alloc));
            std::vector<Tensor> output_tensors;
            execute(input_tensors, output_tensors);

            std::vector<Tensor> model_result_ ;
            // Split output_tensors from [2, 4, 64, 64] to [1, 4, 64, 64]
            Tensor &noise_pred_img_ = model_result_[0];
            Tensor &noise_pred_txt_ = model_result_[1];
            Tensor guided_pred_ = TensorHelper::guidance(
                noise_pred_img_, noise_pred_txt_, sd_unet_config.sd_scale_guidance
            );
        }

        Tensor guided_pred_ = TensorHelper::guidance(
            guided_pred_positive_, guided_pred_negative_, 0.9 /*get from */
        );

        latents_ = sd_scheduler_p->step(model_latent_, guided_pred_, i);
    }

    return latents_;
}


} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_UNET_H

