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

/* ONNXRuntime engine Settings ============================================*/

typedef Ort::Session* OrtSession;
typedef Ort::SessionOptions  OrtOptionConfig;
typedef struct ORTBasicsConfig {
    ExecutionMode          onnx_execution_mode = ExecutionMode::ORT_PARALLEL;
    GraphOptimizationLevel onnx_graph_optimize = GraphOptimizationLevel::ORT_ENABLE_ALL;
} ORTBasicsConfig;


/* Diffusion Scheduler Settings ===========================================*/
/* Scheduler Type Provide */
enum SchedulerType {
    SCHEDULER_EULAR_A           = 1,
    SCHEDULER_LMS               = 2,
};

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

/* Key State & Assistant Const ===========================================*/
/* Model Type */

} // namespace base
} // namespace sd
} // namespace onnx

#endif  // ONNX_SD_CORE_CONFIG_ONCE
