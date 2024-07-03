/*
 * Copyright (c) 2018-2050 ORT_SD_Context Interface - Arikan.Li
 * Created by Arikan.Li on 2024/05/22.
 */
#ifndef ORT_SD_CONTEXT_IMPLEMENT_
#define ORT_SD_CONTEXT_IMPLEMENT_

#include "ort_sd_context.cc"

#include "ort_sd.h"

namespace ortsd {
    ORT_ENTRY void generate_context(IOrtSDContext_ptr *ctx_pp_, struct IOrtSDConfig ctx_config_) {
        // If you have any initial checking logic, plz put in there
        if (ctx_pp_ && *ctx_pp_) return;
        *ctx_pp_ = new onnx::sd::context::OrtSD_Context(
            onnx::sd::context::OrtSD_Config{
                DEFAULT_EXECUTOR_CONFIG,
                {
                    std::string(ctx_config_.sd_modelpath_config.onnx_clip_path),
                    std::string(ctx_config_.sd_modelpath_config.onnx_unet_path),
                    std::string(ctx_config_.sd_modelpath_config.onnx_vae_encoder_path),
                    std::string(ctx_config_.sd_modelpath_config.onnx_vae_decoder_path),
                    std::string(ctx_config_.sd_modelpath_config.onnx_control_net_path),
                    std::string(ctx_config_.sd_modelpath_config.onnx_safty_path),
                },
                {
                    onnx::sd::base::SchedulerType(ctx_config_.sd_scheduler_config.sd_scheduler_type),
                    ctx_config_.sd_scheduler_config.scheduler_training_steps,
                    ctx_config_.sd_scheduler_config.scheduler_beta_start,
                    ctx_config_.sd_scheduler_config.scheduler_beta_end,
                    ctx_config_.sd_scheduler_config.scheduler_seed,
                    onnx::sd::base::BetaType(ctx_config_.sd_scheduler_config.scheduler_beta_type),
                    onnx::sd::base::AlphaType(ctx_config_.sd_scheduler_config.scheduler_alpha_type),
                    onnx::sd::base::PredictionType(ctx_config_.sd_scheduler_config.scheduler_predict_type)
                },
                {
                    onnx::sd::base::TokenizerType(ctx_config_.sd_tokenizer_config.sd_tokenizer_type),
                    ctx_config_.sd_tokenizer_config.tokenizer_dictionary_at,
                    ctx_config_.sd_tokenizer_config.tokenizer_aggregates_at,
                    ctx_config_.sd_tokenizer_config.avail_token_count,
                    ctx_config_.sd_tokenizer_config.avail_token_size,
                    ctx_config_.sd_tokenizer_config.major_hidden_dim,
                    ctx_config_.sd_tokenizer_config.major_boundary_factor,
                    ctx_config_.sd_tokenizer_config.txt_attn_increase_factor,
                    ctx_config_.sd_tokenizer_config.txt_attn_decrease_factor
                },
                ctx_config_.sd_inference_steps,
                ctx_config_.sd_input_width,
                ctx_config_.sd_input_height,
                ctx_config_.sd_input_channel,
                ctx_config_.sd_scale_guidance,
                ctx_config_.sd_decode_scale_strength
            }
        );
    }

    ORT_ENTRY void released_context(IOrtSDContext_ptr *ctx_pp_) {
        if (ctx_pp_ && *ctx_pp_) {
            // ((onnx::sd::context::OrtSD_Context *) *ctx_pp_)->~OrtSD_Context();
            delete ((onnx::sd::context::OrtSD_Context *) *ctx_pp_);
            *ctx_pp_ = nullptr;
        }
    }

    ORT_ENTRY void init(IOrtSDContext_ptr ctx_p_) {
        if (ctx_p_) {
            ((onnx::sd::context::OrtSD_Context *) ctx_p_)->init();
        }
    }

    ORT_ENTRY void prepare(IOrtSDContext_ptr ctx_p_, const char *positive_prompts_, const char *negative_prompts_) {
        if (ctx_p_) {
            ((onnx::sd::context::OrtSD_Context *) ctx_p_)->prepare(
                std::string(positive_prompts_),
                std::string(negative_prompts_)
            );
        }
    }

    ORT_ENTRY IO_IMAGE inference(IOrtSDContext_ptr ctx_p_, IO_IMAGE image_data_) {
        if (ctx_p_) {
            auto result_ = ((onnx::sd::context::OrtSD_Context *) ctx_p_)->inference(
                {
                    image_data_.data_,
                    image_data_.size_
                }
            );
            return {result_.data_, result_.size_};
        }
        return image_data_;
    }

    ORT_ENTRY void release(IOrtSDContext_ptr ctx_p_) {
        if (ctx_p_) {
            ((onnx::sd::context::OrtSD_Context *) ctx_p_)->release();
        }
    }
}

#endif  // ORT_SD_CONTEXT_IMPLEMENT_