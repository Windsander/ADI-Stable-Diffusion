/*
 * Copyright (c) 2018-2050 SD_UNet_Video - Arikan.Li
 * Created by Arikan.Li on 2026/08/26.
 *
 * UNetSpatioTemporalConditionModel (SVD img2vid backbone), batch=1 with
 * dual-call classifier-free guidance. Kept separate from the SD UNet unit:
 * the 4-input signature collision (MMDiT also declares 4 inputs) is avoided
 * entirely, and the video path owns its own scheduler/loop semantics.
 *
 * Loop verified against diffusers 0.39 StableVideoDiffusionPipeline:
 *   latents     = randn * init_noise_sigma                      (scheduler mask)
 *   per step:
 *     scaled    = scale_model_input(latents)                    ( / sqrt(sigma^2+1))
 *     input     = cat([scaled, image_latents], dim=channel)     ( [1,F,8,h,w] )
 *     cond      = unet(input, t, image_embeds,  added_time_ids)
 *     uncond    = unet(input, t, zero_embeds,   added_time_ids) with ZERO image latents
 *     guided    = uncond + ramp(f) * (cond - uncond)            ( linspace(1, max, F) )
 *     latents   = euler_v_step(latents, guided)
 *   added_time_ids = [fps - 1, motion_bucket_id, noise_aug_strength]
 */
#ifndef MODEL_UNET_VIDEO_H
#define MODEL_UNET_VIDEO_H

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

#define DEFAULT_UNET_VIDEO_CONFIG                                    \
    {                                                                \
        /*sd_scheduler_config*/     DEFAULT_SCHEDULER_CONFIG,        \
        /*sd_inference_steps*/      25,                              \
        /*sd_video_frames*/         14,                              \
        /*sd_input_width*/          128,                             \
        /*sd_input_height*/         72,                              \
        /*sd_video_fps*/            7,                               \
        /*sd_video_motion_bucket*/  127,                             \
        /*sd_video_noise_aug*/      0.02f,                           \
        /*sd_guidance_min*/         1.0f,                            \
        /*sd_guidance_max*/         3.0f                             \
    }

typedef struct ModelUNetVideoConfig {
    SchedulerConfig sd_scheduler_config;
    uint64_t sd_inference_steps;
    uint64_t sd_video_frames;
    uint64_t sd_input_width;        // latent width  (pixel / 8)
    uint64_t sd_input_height;       // latent height (pixel / 8)
    uint64_t sd_video_fps;
    uint64_t sd_video_motion_bucket;
    float sd_video_noise_aug;
    float sd_guidance_min;
    float sd_guidance_max;
} ModelUNetVideoConfig;

class UNetVideo : public ModelBase {
private:
    static constexpr int64_t LATENT_C = 4;

    ModelUNetVideoConfig sd_unet_config = DEFAULT_UNET_VIDEO_CONFIG;
    SchedulerEntity_ptr sd_scheduler_p;

protected:
    void generate_output(std::vector<Tensor>& output_tensors_) override {
        int64_t f_ = int64_t(sd_unet_config.sd_video_frames);
        int64_t h_ = int64_t(sd_unet_config.sd_input_height);
        int64_t w_ = int64_t(sd_unet_config.sd_input_width);
        std::vector<float> output_hidden_(f_ * LATENT_C * h_ * w_, 0.0f);
        output_tensors_.emplace_back(TensorHelper::create(
            TensorShape{1, f_, LATENT_C, h_, w_}, output_hidden_
        ));
    }

public:
    explicit UNetVideo(const std::string &model_path_,
                       const ModelUNetVideoConfig &unet_config_ = DEFAULT_UNET_VIDEO_CONFIG)
        : ModelBase(model_path_) {
        sd_unet_config = unet_config_;
        sd_scheduler_p = SchedulerRegister::request_scheduler(unet_config_.sd_scheduler_config);
    }

    ~UNetVideo() override {
        sd_scheduler_p = SchedulerRegister::recycle_scheduler(sd_scheduler_p);
        sd_unet_config.~ModelUNetVideoConfig();
    }

    // image_embeds_: [1, 1, 1024] CLIP image embedding
    // image_latents_: [1, 4, h, w] deterministic VAE mean (SVD: NOT scaled)
    // returns denoised latents [1, F, 4, h, w]
    Tensor inference(const Tensor &image_embeds_, const Tensor &image_latents_) {
        int64_t f_ = int64_t(sd_unet_config.sd_video_frames);
        int64_t h_ = int64_t(sd_unet_config.sd_input_height);
        int64_t w_ = int64_t(sd_unet_config.sd_input_width);
        const uint64_t working_steps_ = sd_scheduler_p->init(sd_unet_config.sd_inference_steps);

        // adapt the timestep tensor to the export's declared signature
        // (fp32 scalar {1} from our export; tolerate int64 legacy-style)
        ONNXTensorElementDataType timestep_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        size_t timestep_rank_ = 1;
        for (size_t ii = 0; ii < model_input_count(); ++ii) {
            const std::string n_ = model_input_name(ii);
            if (n_ == "timestep" || n_ == "t") {
                ONNXTensorElementDataType declared_type_ = model_input_element_type(ii, &timestep_rank_);
                if (declared_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
                    timestep_type_ = declared_type_;
                }
                break;
            }
        }

        // constant conditioning tensors
        Tensor zero_embeds_ = TensorHelper::create(
            TensorShape{1, 1, 1024}, std::vector<float>(1024, 0.0f)
        );
        Tensor image_frames_ = TensorHelper::repeat_frames<float>(image_latents_, f_);   // [1,F,4,h,w]
        Tensor zero_frames_ = TensorHelper::create(
            TensorShape{1, f_, LATENT_C, h_, w_}, std::vector<float>(f_ * LATENT_C * h_ * w_, 0.0f)
        );
        // diffusers: SVD was conditioned on fps-1 during training
        std::vector<float> time_ids_value_ = {
            float(sd_unet_config.sd_video_fps > 0 ? sd_unet_config.sd_video_fps - 1 : 0),
            float(sd_unet_config.sd_video_motion_bucket),
            sd_unet_config.sd_video_noise_aug
        };
        Tensor added_time_ids_ = TensorHelper::create(TensorShape{1, 3}, time_ids_value_);

        // initial noise latents [1, F, 4, h, w] scaled by init_noise_sigma
        Tensor latents_ = sd_scheduler_p->mask(TensorShape{1, f_, LATENT_C, h_, w_});

        const bool need_guidance_ = (sd_unet_config.sd_guidance_max > 1.0f);

        auto bind_inputs_ = [&](const Tensor& lat8_, const Tensor& ts_,
                                const Tensor& embs_) {
            std::vector<Tensor> inputs_;
            for (size_t ii = 0; ii < model_input_count(); ++ii) {
                const std::string n_ = model_input_name(ii);
                if (n_ == "sample" || n_ == "hidden_states" || n_ == "latent_model_input") {
                    inputs_.emplace_back(TensorHelper::clone<float_t>(lat8_));
                } else if (n_ == "encoder_hidden_states") {
                    inputs_.emplace_back(TensorHelper::clone<float_t>(embs_));
                } else if (n_ == "added_time_ids" || n_ == "time_ids") {
                    inputs_.emplace_back(TensorHelper::clone<float_t>(added_time_ids_));
                } else if (n_ == "timestep" || n_ == "t") {
                    auto ts_type_ = ts_.GetTensorTypeAndShapeInfo().GetElementType();
                    float v_ = (ts_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ?
                               ts_.GetTensorData<float>()[0] :
                               float(ts_.GetTensorData<int64_t>()[0]);
                    if (timestep_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                        TensorShape ts_shape_ = (timestep_rank_ == 0) ? TensorShape{} : TensorShape{1};
                        inputs_.emplace_back(TensorHelper::create<float>(ts_shape_, std::vector<float>{v_}));
                    } else {
                        inputs_.emplace_back(TensorHelper::create<int64_t>(TensorShape{1}, std::vector<int64_t>{int64_t(std::llround(v_))}));
                    }
                } else {
                    amon_exception(basic_exception(EXC_LOG_ERR, "ERROR:: unbound SVD UNet model input"));
                }
            }
            return inputs_;
        };

        for (int i = 0; i < working_steps_; ++i) {
            Tensor scaled_ = sd_scheduler_p->scale(latents_, i);   // [1,F,4,h,w]
            Tensor timestep_ = sd_scheduler_p->time(i);            // float {1}

            // positive (conditioned) pass
            Tensor input_cond_ = TensorHelper::concat_frame_channels<float>(scaled_, image_frames_);
            std::vector<Tensor> inputs_cond_ = bind_inputs_(input_cond_, timestep_, image_embeds_);
            std::vector<Tensor> outputs_cond_;
            generate_output(outputs_cond_);
            execute(inputs_cond_, outputs_cond_);
            Tensor pred_positive_ = std::move(outputs_cond_.front());

            if (!need_guidance_) {
                latents_ = sd_scheduler_p->step(latents_, pred_positive_, i, 1.0f);
                CommonHelper::print_progress_bar(float(i + 1) / float(working_steps_));
                continue;
            }

            // negative (unconditioned) pass: zero embeds AND zero image latents
            Tensor input_uncond_ = TensorHelper::concat_frame_channels<float>(scaled_, zero_frames_);
            std::vector<Tensor> inputs_uncond_ = bind_inputs_(input_uncond_, timestep_, zero_embeds_);
            std::vector<Tensor> outputs_uncond_;
            generate_output(outputs_uncond_);
            execute(inputs_uncond_, outputs_uncond_);
            Tensor pred_negative_ = std::move(outputs_uncond_.front());

            // per-frame guidance ramp linspace(min, max, F)
            Tensor guided_pred_ = TensorHelper::guide_frames<float>(
                pred_negative_, pred_positive_,
                sd_unet_config.sd_guidance_min, sd_unet_config.sd_guidance_max
            );

            latents_ = sd_scheduler_p->step(latents_, guided_pred_, i, 1.0f);

            CommonHelper::print_progress_bar(float(i + 1) / float(working_steps_));
        }

        sd_scheduler_p->uninit();
        return latents_;
    }
};

} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_UNET_VIDEO_H
