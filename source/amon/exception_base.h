// Copyright (c) 2018-2050 CoreContext - Arikan.Li

#ifndef ONNX_SD_BASE_EXCEPTION_ONCE
#define ONNX_SD_BASE_EXCEPTION_ONCE

#include <stdexcept>
#include <iostream>
#include "basic_logger.h"

namespace onnx {
namespace sd {
namespace amon {

enum ExceptionLevel {
    EXC_LOG_CRIT = 1,
    EXC_LOG_ERR = 2,
    EXC_LOG_WARN = 3,
    EXC_LOG_INFO = 4,
    EXC_LOG_DEBUG = 5,
    EXC_LOG_VERBOSE = 6
};

class exception_base : public std::exception {
private:
    ExceptionLevel exception_level;
    const char *exception_message;

protected:
    virtual void before_report(ExceptionLevel level_, const char *msg_) = 0;
    virtual void after_report() = 0;

public:
    explicit exception_base(
        ExceptionLevel level_ = EXC_LOG_INFO,
        const char *message_ = "") : std::exception() {
        exception_level = level_;
        exception_message = message_;
    }

    ~exception_base() override = default;

    ExceptionLevel level() const {
        return exception_level;
    }

    const char *what() const noexcept override {
        return exception_message;
    }

    void report() {
        before_report(exception_level, exception_message);
        sd_log(((loglevel_e) exception_level)) << exception_message;
        after_report();
    }
};

} // namespace amon
} // namespace sd
} // namespace onnx

#endif // ONNX_SD_BASE_EXCEPTION_ONCE