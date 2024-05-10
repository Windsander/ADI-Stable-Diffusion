// Copyright (c) 2018-2050 CoreContext - Arikan.Li

#include "exception_base.h"

namespace onnx {
namespace sd {
namespace amon {

class class_exception : public exception_base {
private:
  void before_report(ExceptionLevel level_, const char* msg_) override
  {
    LOG_PARAMS_UNUSED(level_);
    LOG_PARAMS_UNUSED(msg_);
  }

  void after_report() override
  {

  }

public:
  explicit class_exception(ExceptionLevel level_, const char* message_ = "") : exception_base(level_, message_)
  {
    // no-action
  }

  ~class_exception() override = default;
};

} // namespace amon
} // namespace sd
} // namespace onnx
