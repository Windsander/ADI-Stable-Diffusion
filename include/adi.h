﻿/*
 * Copyright (c) 2018-2050 ORT_SD_Context Interface - Arikan.Li
 * Created by Arikan.Li on 2024/05/22.
 */
#ifndef ORT_SD_CONTEXT_H_
#define ORT_SD_CONTEXT_H_

#define ORT_ENTRY

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define CURRENT_ADI_VERSION "v1.0.1"

/* Inference Execution Settings ===========================================*/

/* Executor Type */
enum AvailableExecutionType {
    EXECUTOR_CPU               = 0x00,
    EXECUTOR_GPU_AUTO          = 0x01,
    EXECUTOR_GPU_COREML        = 0x02,
    EXECUTOR_GPU_TENSORRT      = 0x03,
    EXECUTOR_GPU_CUDA          = 0x04,
    EXECUTOR_GPU_NNAPI         = 0x05,
    AVAILABLE_EXECUTOR_COUNT,
};

/* Diffusion Abilities Settings ===========================================*/

/* Scheduler Beta Provide */
enum AvailableBetaType {
    BETA_TYPE_LINEAR            = 0x00,
    BETA_TYPE_SCALED_LINEAR     = 0x01,
    BETA_TYPE_SQUAREDCOS_CAP_V2 = 0x02,
    BETA_COUNT,
};

/* Scheduler Alpha Provide */
enum AvailableAlphaType {
    ALPHA_TYPE_COSINE           = 0x00,
    ALPHA_TYPE_EXP              = 0x01,
    ALPHA_COUNT,
};

/* Scheduler Prediction Provide */
enum AvailablePredictionType {
    PREDICT_TYPE_EPSILON        = 0x00,
    PREDICT_TYPE_V_PREDICTION   = 0x01,
    PREDICT_TYPE_SAMPLE         = 0x02,
    AVAILABLE_PREDICTOR_COUNT,
};

/* Scheduler Type Provide */
enum AvailableSchedulerType {
    AVAILABLE_SCHEDULER_EULER       = 0x00,
    AVAILABLE_SCHEDULER_EULER_A     = 0x01,
    AVAILABLE_SCHEDULER_LMS         = 0x02,
    AVAILABLE_SCHEDULER_LCM         = 0x03,
    AVAILABLE_SCHEDULER_HEUN        = 0x04,
    AVAILABLE_SCHEDULER_DDPM        = 0x05,
    AVAILABLE_SCHEDULER_DDIM        = 0x06,
    AVAILABLE_SCHEDULER_UNIPC       = 0x07,
    AVAILABLE_SCHEDULER_COUNT,
};

/* Tokenizer Type Provide */
enum AvailableTokenizerType {
    AVAILABLE_TOKENIZER_BPE         = 0x00,
    /*AVAILABLE_TOKENIZER_WORD_PIECE  = 0x02,*/
    AVAILABLE_TOKENIZER_COUNT,
};

/* Diffusion Main Configuration ===========================================*/
/* OrtSD Context IO data struct*/
typedef struct IO_IMAGE {
    uint8_t *data_;
    uint64_t size_;
} IO_IMAGE;

/**
 * @details OrtSD Context functional params
 *
 * @example see DEFAULT_SDXL_CONFIG
 */
typedef struct IOrtSDConfig {
    enum AvailableExecutionType sd_executor_type;   // Base: choose inference executor (default: CPU)
    struct {
        const char* onnx_clip_path;                 // Model: CLIP Path (also known as text_encoder)
        const char* onnx_unet_path;                 // Model: UNet Path
        const char* onnx_vae_encoder_path;          // Model: VAE Encoder Path (also known as vae_encoder)
        const char* onnx_vae_decoder_path;          // Model: VAE Decoder Path (also known as vae_decoder)
        const char* onnx_control_net_path;          // Model: ControlNet Path (currently not available)
        const char* onnx_safty_path;                // Model: Safety Security Model Path (currently not available)
    } sd_modelpath_config;

    struct {
        enum AvailableSchedulerType sd_scheduler_type;  // Scheduler: scheduler type (Euler_A, LMS, ... etc.)
        uint64_t scheduler_training_steps;              // Scheduler: scheduler steps when at model training stage (can be found in model details, set by manual)
        uint64_t scheduler_maintain_cache;              // Scheduler: scheduler maintain history records count (only when scheduler type using, control by self )
        float scheduler_beta_start;                     // Scheduler: Beta start (recommend 0.00085f)
        float scheduler_beta_end;                       // Scheduler: Beta end (recommend 0.012f)
        int64_t scheduler_seed;                         // Scheduler: seed for random (config if u need certain output)
        enum AvailableBetaType scheduler_beta_type;     // Scheduler: Beta Style (Linear. ScaleLinear, CAP_V2)
        enum AvailableAlphaType scheduler_alpha_type;   // Scheduler: Alpha(Beta) Method (Cos, Exp)
        enum AvailablePredictionType scheduler_predict_type;   // Scheduler: Prediction Style (Epsilon, V_Pred, Sample)
    } sd_scheduler_config;

    struct {
        enum AvailableTokenizerType sd_tokenizer_type;  // Tokenizer: tokenizer type (currently only provide BPE)
        const char* tokenizer_dictionary_at;        // Tokenizer: vocabulary lib <one vocab per line, row treate as index>
        const char* tokenizer_aggregates_at;        // Tokenizer: merges file <one merge-pair per line, currently only for BPE>
        int32_t avail_token_count;                  // Tokenizer: all available token in vocabulary totally
        int32_t avail_token_size;                   // Tokenizer: max token length (include <start> & <end>, so 75 avail)
        int32_t major_hidden_dim;                   // Tokenizer: out token length
        float major_boundary_factor;                // Tokenizer: weights for <start> & <end> mark-token
        float txt_attn_increase_factor;             // Tokenizer: weights for (prompt) to gain attention by this factor
        float txt_attn_decrease_factor;             // Tokenizer: weights for [prompt] to loss attention by this factor
    } sd_tokenizer_config;

    uint64_t sd_inference_steps;            // Infer_Major: inference step
    uint64_t sd_input_width;                // Infer_Major: IO image width (match SD-model training sets, Constant)
    uint64_t sd_input_height;               // Infer_Major: IO image height (match SD-model training sets, Constant)
    uint64_t sd_input_channel;              // Infer_Major: IO image channel (match SD-model training sets, Constant)
    float sd_scale_guidance;                // Infer_Major: immersion rate for [value * (Positive - Negative)] residual
    float sd_random_intensity;              // Infer_Major: random intensity for in stepping noise Add (only avail when method supported)
    float sd_decode_scale_strength;         // Infer_Major: for VAE Decoding result merged (Recommend 0.18215f)
} IOrtSDConfig;

namespace ortsd{
    typedef void* IOrtSDContext_ptr;

    ORT_ENTRY void generate_context(IOrtSDContext_ptr* ctx_pp_, struct IOrtSDConfig ctx_config_);
    ORT_ENTRY void released_context(IOrtSDContext_ptr* ctx_pp_);
    ORT_ENTRY void init(IOrtSDContext_ptr ctx_p_);
    ORT_ENTRY void prepare(IOrtSDContext_ptr ctx_p_, const char* positive_prompts_, const char*negative_prompts_);
    ORT_ENTRY IO_IMAGE inference(IOrtSDContext_ptr ctx_p_, IO_IMAGE image_data_);
    ORT_ENTRY void release(IOrtSDContext_ptr ctx_p_);
}

#ifdef __cplusplus
}
#endif

#endif  // ORT_SD_CONTEXT_H_