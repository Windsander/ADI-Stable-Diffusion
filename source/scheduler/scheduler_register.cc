/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_REGISTER_ONCE
#define SCHEDULER_REGISTER_ONCE

#include "onnxsd_foundation.cc"
#include "scheduler_base.cc"
#include "scheduler_discrete_eular_a.cc"
#include "scheduler_discrete_lms.cc"

namespace onnx {
namespace sd {
namespace scheduler {

using namespace base;

typedef SchedulerBase SchedulerEntity;
typedef SchedulerBase* SchedulerEntity_ptr;

class SchedulerRegister {
public:
    static SchedulerEntity_ptr request_scheduler(SchedulerType scheduler_type, const SchedulerConfig &scheduler_config_) {
        SchedulerEntity_ptr result_ptr_ = nullptr;
        switch (scheduler_type) {
            case SCHEDULER_EULAR_A: {
                result_ptr_ = new EularAncestralDiscreteScheduler(scheduler_config_);
                break;
            }
            case SCHEDULER_LMS: {
                result_ptr_ = new LMSDiscreteScheduler(scheduler_config_);
                break;
            }
            default:{
                amon_report(class_exception(EXC_LOG_ERR, "ERROR:: selected Scheduler unimplemented"));
                break;
            }
        }
        if (result_ptr_){
            result_ptr_->create();
        }
        return result_ptr_;
    }

    static SchedulerEntity_ptr recycle_scheduler(SchedulerEntity_ptr target_ptr_){
        if (target_ptr_){
            target_ptr_->release();
            delete target_ptr_;
        }
        return nullptr;
    }
};

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif  // SCHEDULER_REGISTER_ONCE