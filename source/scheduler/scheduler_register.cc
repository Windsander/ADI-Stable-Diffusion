/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_REGISTER_ONCE
#define SCHEDULER_REGISTER_ONCE

#include "scheduler_base.cc"
#include "scheduler_eular_a_discrete.cc"
#include "scheduler_lcm_karras.cc"


namespace onnx {
namespace sd {
namespace scheduler {

typedef SchedulerBase SchedulerBase;
typedef SchedulerBase* SchedulerBase_ptr;

class SchedulerRegister : public EntityRegister<SchedulerRegister, SchedulerBase> {
public:
  void regist_all_generator() override
  {
    signin_generator<RenderEnvD2D3D10>(RenderDriverType::DRIVER_TYPE_D2D3D10);
    signin_generator<RenderEnvD2D3D11>(RenderDriverType::DRIVER_TYPE_D2D3D11);
    signin_generator<RenderEnvD2D>(RenderDriverType::DRIVER_TYPE_D2D);
    signin_generator<RenderEnvDX9>(RenderDriverType::DRIVER_TYPE_Dx9);
    signin_generator<RenderEnvDX10>(RenderDriverType::DRIVER_TYPE_Dx10);
    signin_generator<RenderEnvDX11>(RenderDriverType::DRIVER_TYPE_Dx11);
  }
};

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif  // SCHEDULER_REGISTER_ONCE