// Copyright (c) 2018-2050 CoreContext - Arikan.Li

#ifndef ONNX_SD_EXCEPTION_ENTRY_ONCE
#define ONNX_SD_EXCEPTION_ENTRY_ONCE

#include "cens_statistics_base.h"
#include "cens_target_fps.h"

#include "exception_base.h"
#include "exception_class.h"
#include "exception_register.h"

#ifndef render_exception
#define render_exception(e) \
  {                         \
    e.report();             \
    throw e;                \
  }
#endif

#ifndef render_report
#define render_report(e)    \
  {                         \
    e.report();             \
  }
#endif

#endif // ONNX_SD_EXCEPTION_ENTRY_ONCE