// Copyright (c) 2018-2050 CoreContext - Arikan.Li

#ifndef ONNX_SD_EXCEPTION_ENTRY_ONCE
#define ONNX_SD_EXCEPTION_ENTRY_ONCE

#include "cens_statistics_base.h"
#include "cens_target_fps.h"

#include "exception_base.h"
#include "exception_type.h"

#ifndef amon_exception
#define amon_exception(e)   \
  {                         \
    e.report();             \
    throw e;                \
  }
#endif

#ifndef amon_report
#define amon_report(e)      \
  {                         \
    e.report();             \
  }
#endif

#endif // ONNX_SD_EXCEPTION_ENTRY_ONCE