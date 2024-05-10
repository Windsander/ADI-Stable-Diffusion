/*
 * Copyright (c) 2018-2050 BasicDefs&Refs - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
*/
#ifndef ONNX_SD_DEFS_H_
#define ONNX_SD_DEFS_H_

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <napi.h>

#include <string>
#include <algorithm>
#include <functional>
#include <map>
#include <cmath>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "onnxruntime_cxx_api.h"
#include "exceptions_entry.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#endif
#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#endif
#ifdef USE_COREML
#include "core/providers/coreml/coreml_provider_factory.h"
#endif

#endif
