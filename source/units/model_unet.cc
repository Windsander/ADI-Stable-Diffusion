/*
 * Copyright (c) 2018-2050 SD_UNet - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_UNET_H
#define MODEL_UNET_H

#include "model_base.cc"
#include "scheduler_register.cc"

namespace onnx {
namespace sd {
namespace units {

using namespace base;
using namespace amon;
using namespace scheduler;
using namespace Ort;
using namespace detail;

#define DEFAULT_UNET_CONFIG                                          \
    {                                                                \
        /*sd_scheduler_config*/ DEFAULT_SCHEDULER_CONFIG,            \
        /*sd_inference_steps*/  3,                                   \
        /*sd_input_width*/      512,                                 \
        /*sd_input_height*/     512,                                 \
        /*sd_input_channel*/    4,                                   \
        /*sd_scale_guidance*/   7.5f,                                \
        /*sd_random_intensity*/ 1.0f                                 \
    }                                                                \

typedef struct ModelUNetConfig {
    SchedulerConfig sd_scheduler_config;
    uint64_t sd_inference_steps;
    uint64_t sd_input_width;
    uint64_t sd_input_height;
    uint64_t sd_input_channel;
    float sd_scale_guidance;
    float sd_random_intensity;
} ModelUNetConfig;

class UNet : public ModelBase {
private:
    ModelUNetConfig sd_unet_config = DEFAULT_UNET_CONFIG;
    SchedulerEntity_ptr sd_scheduler_p;

protected:
    void generate_output(std::vector<Tensor>& output_tensors_) override;

public:
    explicit UNet(const std::string &model_path_, const ModelUNetConfig &unet_config_ = DEFAULT_UNET_CONFIG);
    ~UNet() override;

    Tensor inference(const Tensor &embs_positive_,const Tensor &embs_negative_, const Tensor &encoded_img_);
};

UNet::UNet(const std::string &model_path_, const ModelUNetConfig& unet_config_) : ModelBase(model_path_){
    sd_unet_config = unet_config_;
    sd_scheduler_p = SchedulerRegister::request_scheduler(unet_config_.sd_scheduler_config);
}

UNet::~UNet(){
    sd_scheduler_p = SchedulerRegister::recycle_scheduler(sd_scheduler_p);
    sd_unet_config.~ModelUNetConfig();
}

void UNet::generate_output(std::vector<Tensor> &output_tensors_) {
    std::vector<float> output_hidden_(
        sd_unet_config.sd_input_width *
        sd_unet_config.sd_input_height *
        sd_unet_config.sd_input_channel, 0.0f
    );
    TensorShape hidden_shape_ = {
        1,
        int64_t(sd_unet_config.sd_input_channel),
        int64_t(sd_unet_config.sd_input_height),
        int64_t(sd_unet_config.sd_input_width)
    };
    output_tensors_.emplace_back(TensorHelper::create(hidden_shape_, output_hidden_));
}

Tensor UNet::inference(
    const Tensor &embs_positive_,
    const Tensor &embs_negative_,
    const Tensor &encoded_img_
) {
    int w_ = int(sd_unet_config.sd_input_width);
    int h_ = int(sd_unet_config.sd_input_height);
    int c_ = int(sd_unet_config.sd_input_channel);
    const bool need_guidance_ = (sd_unet_config.sd_scale_guidance > 1);
    const uint64_t working_steps_ = sd_scheduler_p->init(sd_unet_config.sd_inference_steps);

    TensorShape latent_shape_{1, c_, h_, w_};
    std::vector<float> latent_empty_(c_ * h_ * w_, 0.0f);
    Tensor latents_ = (TensorHelper::have_data(encoded_img_)) ?
                      TensorHelper::clone<float>(encoded_img_, latent_shape_) :
                      TensorHelper::create(latent_shape_, latent_empty_);
    Tensor init_mask_ = sd_scheduler_p->mask(latent_shape_);
    latents_ = TensorHelper::add<float>(latents_, init_mask_, latent_shape_);

    for (int i = 0; i < working_steps_; ++i) {
        Tensor model_latent_ = sd_scheduler_p->scale(latents_, i);
        Tensor timestep_ = sd_scheduler_p->time(i);

        // do positive N_pos_embed_num times
        Tensor pred_positive_ = TensorHelper::create(TensorShape{0}, std::vector<float>{});
        if (TensorHelper::have_data(embs_positive_)) {
            std::vector<Tensor> input_tensors;
            input_tensors.emplace_back(TensorHelper::clone<float_t>(model_latent_));
            input_tensors.emplace_back(TensorHelper::clone<int64_t>(timestep_));
            input_tensors.emplace_back(TensorHelper::clone<float_t>(embs_positive_));
            std::vector<Tensor> output_tensors;
            generate_output(output_tensors);
            execute(input_tensors, output_tensors);
            pred_positive_ = std::move(output_tensors[0]);
        }

        // do negative N_neg_embed_num times
        Tensor pred_negative_ = TensorHelper::create(TensorShape{0}, std::vector<float>{});
        if (TensorHelper::have_data(embs_negative_) && need_guidance_) {
            std::vector<Tensor> input_tensors;
            input_tensors.emplace_back(TensorHelper::clone<float_t>(model_latent_));
            input_tensors.emplace_back(TensorHelper::clone<int64_t>(timestep_));
            input_tensors.emplace_back(TensorHelper::clone<float_t>(embs_negative_));
            std::vector<Tensor> output_tensors;
            generate_output(output_tensors);
            execute(input_tensors, output_tensors);
            pred_negative_ = std::move(output_tensors[0]);
        }

        // Merge predictions
        float merge_factor_ = sd_unet_config.sd_scale_guidance;
        Tensor guided_pred_ = (
            (need_guidance_) ?
            TensorHelper::guide<float>(pred_negative_, pred_positive_, merge_factor_) :
            TensorHelper::clone<float>(pred_positive_, latent_shape_)
        );

        // Dnoise & Step
        latents_ = sd_scheduler_p->step(latents_, guided_pred_, i, sd_unet_config.sd_random_intensity);

        CommonHelper::print_progress_bar(float(i + 1) / float(working_steps_));
    }

    sd_scheduler_p->uninit();
    return latents_;
}


} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_UNET_H

