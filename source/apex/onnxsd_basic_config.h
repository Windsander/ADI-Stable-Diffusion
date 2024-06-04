/*
 * ORT Stable-Diffusion Basic Core Config 基础
 * Definition: Stable-Diffusion 的过程中基础能力配置
 *
 * Created by Arikan.Li on 2024/05/10.
 */
#ifndef ONNX_SD_CORE_CONFIG_ONCE
#define ONNX_SD_CORE_CONFIG_ONCE

#include "onnxsd_base_defs.h"

namespace onnx {
namespace sd {
namespace base {

#define DEFAULT_ORT_ENGINE_NAME "OrtSD-Engine"

/* Basement Data Type =====================================================*/
/* IO data*/
typedef uint8_t IMAGE_BYTE;
typedef uint64_t IMAGE_SIZE;
typedef struct IMAGE_DATA {
    uint8_t *data_;
    uint64_t size_;
} IMAGE_DATA;

/* ONNXRuntime engine Settings ============================================*/

typedef Ort::Value Tensor;
typedef std::vector<int64_t> TensorShape;

typedef Ort::Session* OrtSession;
typedef Ort::SessionOptions  OrtOptionConfig;

#define DEFAULT_EXECUTOR_CONFIG                                         \
    {                                                                   \
        /*onnx_execution_mode*/ ExecutionMode::ORT_PARALLEL,            \
        /*onnx_graph_optimize*/ GraphOptimizationLevel::ORT_ENABLE_ALL  \
    }

typedef struct ORTBasicsConfig {
    ExecutionMode          onnx_execution_mode;
    GraphOptimizationLevel onnx_graph_optimize;
} ORTBasicsConfig;

/* Diffusion Scheduler Settings ===========================================*/
/* Scheduler Type Provide */
typedef enum SchedulerType {
    SCHEDULER_EULER             = 1,
    SCHEDULER_EULER_A           = 2,
    SCHEDULER_LMS               = 3,
} SchedulerType;

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

#define DEFAULT_SCHEDULER_CONFIG                             \
    {                                                        \
        /*scheduler_type*/              SCHEDULER_EULER_A,   \
        /*scheduler_training_steps*/    1000,                \
        /*scheduler_beta_start*/        0.00085f,            \
        /*scheduler_beta_end*/          0.012f,              \
        /*scheduler_seed*/              42,                  \
        /*scheduler_beta_type*/         BETA_TYPE_LINEAR,    \
        /*scheduler_alpha_type*/        ALPHA_TYPE_COSINE,   \
        /*scheduler_predict_type*/      PREDICT_TYPE_EPSILON \
    }

typedef struct SchedulerConfig {
    SchedulerType scheduler_type;
    uint64_t scheduler_training_steps;
    float scheduler_beta_start;
    float scheduler_beta_end;
    int64_t scheduler_seed;
    BetaType scheduler_beta_type;
    AlphaType scheduler_alpha_type;
    PredictionType scheduler_predict_type;
} SchedulerConfig;

/* Diffusion Tokenizer Settings ===========================================*/
/* Tokenizer Type Provide */
typedef enum TokenizerType {
    TOKENIZER_BPE               = 1,
    TOKENIZER_WORD_PIECE        = 2,
} TokenizerType;

#define DEFAULT_TOKENIZER_CONFIG                            \
    {                                                       \
         /*tokenizer_type*/              TOKENIZER_BPE,     \
         /*tokenizer_dictionary_at*/     "",                \
         /*avail_token_count*/           49408,             \
         /*avail_token_size*/            77,                \
         /*major_hidden_dim*/            768,               \
         /*major_boundary_factor*/       1.0f,              \
         /*txt_attn_increase_factor*/    1.1f,              \
         /*txt_attn_decrease_factor*/    1 / 1.1f,          \
    }

typedef struct TokenizerConfig {
    TokenizerType tokenizer_type;
    std::string tokenizer_dictionary_at;        // vocabulary lib <one vocab per line, row treate as index>
    int32_t avail_token_count;                  // all available token in vocabulary totally
    int32_t avail_token_size;                   // max token length (include <start> & <end>, so 75 avail)
    int32_t major_hidden_dim;                   // out token length
    float major_boundary_factor;
    float txt_attn_increase_factor;
    float txt_attn_decrease_factor;
} TokenizerConfig;

/* Key State & Assistant Const ===========================================*/
/* Model Type */

} // namespace base
} // namespace sd
} // namespace onnx

#endif  // ONNX_SD_CORE_CONFIG_ONCE
