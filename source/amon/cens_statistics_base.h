// Copyright (c) 2018-2050 CoreContext - Arikan.Li

#ifndef ONNX_SD_BASE_CENS_ONCE
#define ONNX_SD_BASE_CENS_ONCE

#include <stdexcept>
#include <iostream>
#include "basic_logger.h"

namespace onnx {
namespace sd {
namespace amon {

enum StatisticsLevel {
  CENS_LOG_WARN    = 3,
  CENS_LOG_INFO    = 4,
  CENS_LOG_DEBUG   = 5,
  CENS_LOG_VERBOSE = 6
};

class statistics_base {
private:
  StatisticsLevel statistics_level;
  const char* statistics_message;

protected:
  virtual void do_statistics(StatisticsLevel level_, const char* msg_) = 0;

public:
  explicit statistics_base(
      StatisticsLevel level_ = CENS_LOG_INFO,
      const char* message_ = "")
  {
    statistics_level   = level_;
    statistics_message = message_;
  }

  ~statistics_base() = default;

  StatisticsLevel level() const
  {
    return statistics_level;
  }

  const char* what() const noexcept
  {
    return statistics_message;
  }

  void report()
  {
    do_statistics(statistics_level, statistics_message);
  }
};

} // namespace amon
} // namespace sd
} // namespace onnx

#endif // ONNX_SD_BASE_CENS_ONCE