// Copyright (c) 2018-2050 CoreContext - Arikan.Li

#include "exception_base.h"

#define NO_EXTRA_ACTION
#define REGIST_EXCEPTION_TYPE(type_, before_report_, after_report_)                        \
class type_##exception : public exception_base {                                           \
private:                                                                                   \
    void before_report(ExceptionLevel level_, const char *msg_) override {                 \
        LOG_PARAMS_UNUSED(level_);                                                         \
        LOG_PARAMS_UNUSED(msg_);                                                           \
        before_report_                                                                     \
    }                                                                                      \
                                                                                           \
    void after_report() override {                                                         \
        after_report_                                                                      \
    }                                                                                      \
                                                                                           \
public:                                                                                    \
    explicit type_##exception(ExceptionLevel level_, const char *message_ = "") :          \
        exception_base(level_, message_) {                                                 \
    }                                                                                      \
                                                                                           \
    ~type_##exception() override = default;                                                \
}

namespace onnx {
namespace sd {
namespace amon {

    REGIST_EXCEPTION_TYPE(class_    , NO_EXTRA_ACTION, NO_EXTRA_ACTION);
    REGIST_EXCEPTION_TYPE(basic_    , NO_EXTRA_ACTION, NO_EXTRA_ACTION);
    REGIST_EXCEPTION_TYPE(register_ , NO_EXTRA_ACTION, NO_EXTRA_ACTION);

} // namespace amon
} // namespace sd
} // namespace onnx
