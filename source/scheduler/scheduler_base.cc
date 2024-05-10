/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_BASE_H
#define SCHEDULER_BASE_H

#include "onnxsd_defs.h"
#include "onnxsd_basic_register.cc"

namespace onnx {
namespace sd {
namespace scheduler {

using namespace base;
using namespace amon;

/*渲染物体定义===========================================================*/
/* 渲染物体 */
class SchedulerBase {
protected:
    int           train_timesteps_num;
    float         scheduler_init_sigma{};
    vector<int>   scheduler_timesteps;
    vector<float> scheduler_sigmas;


public:
    explicit SchedulerBase(int train_timesteps_num_ = 1000);
    virtual ~SchedulerBase();

    void setTimesteps(int timesteps_);
    int getTimesteps();

    /* 观察关联 */
    void setCurrentObserver(ObjectObserver current_observer_);

    /* 位姿计算 */
    Matrix_4x4f get_convert_matrix(bool use_transpose = true);
    Matrix_4x4f get_presents_matrix();
    Matrix_4x4f get_observer_matrix();
    Matrix_4x4f get_coordinate_matrix();

    /* 生命周期 */
    void init(core_manager_ptr core_driver_, ress_manager_ptr res_pool_);
    void config();
    void add();
    void update();
    void del();
    void release();

    /* 物体绘制 */
    void drawSelf();
    void applyEffect();
    void applyVisible(bool enable);

private:
//  bool has_initialized = false;
};

SchedulerBase::SchedulerBase(int train_timesteps_num_){
    this->train_timesteps_num = train_timesteps_num_;
}


} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_BASE_H
