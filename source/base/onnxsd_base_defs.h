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
#include <stdint.h>

#include <string>
#include <algorithm>
#include <functional>
#include <map>
#include <cmath>
#include <mutex>
#include <vector>
#include <random>
#include <atomic>
#include <sstream>
#include <memory>
#include <unordered_map>

#include "exceptions_entry.h"

#ifdef USE_CUDA
#include "cuda_provider_factory.h"
#endif
#ifdef USE_DML
#include "dml_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include tensorrt_provider_factory.h"
#endif
#ifdef USE_COREML
#include "coreml_provider_factory.h"
#endif

#include "onnxruntime_cxx_api.h"

typedef Ort::Value Tensor;
typedef uint8_t *IMAGE_BYTE;
typedef std::vector<int64_t> TensorShape;

#define SD_UNUSED(x) (void)x

#endif
