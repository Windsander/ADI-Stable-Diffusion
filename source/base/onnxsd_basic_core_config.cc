/*
 * ORT Stable-Diffusion Basic Core Config 基础
 * Definition: Stable-Diffusion 的过程中基础能力配置
 *
 * Created by Arikan.Li on 2024/05/10.
 */
#include "onnxsd_defs.h"

namespace onnx {
namespace sd {
namespace base {

/* Diffusion Scheduler Settings ===========================================*/
/* Scheduler Type Provide */
enum SchedulerType {
    SCHEDULER_EULAR_A           = 1,
    SCHEDULER_LMS               = 2,
};

/* Key State & Assistant Const ===========================================*/
/* Model Type */

} // namespace base
} // namespace sd
} // namespace onnx
