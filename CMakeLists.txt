cmake_minimum_required(VERSION 3.12)
project("onnxruntime-sd")

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SD_STANDALONE ON)
else()
    set(SD_STANDALONE OFF)
endif()


#
# 1. Option list
#
option(ORT_BUILD_EXAMPLES            "ort-sd: build examples" ${SD_STANDALONE})
option(ORT_BUILD_SHARED_LIBS         "ort-sd: build shared libs" OFF)
option(ORT_ONLINE_COMPILED           "ort-sd: using online onnxruntime(ort)" ON)
#option(SD_BUILD_SERVER               "sd: build server example"                           ON)


#
# 2. Add inference engine to project
#
set(ONNX_PATH onnx)
set(ONNX_INFERENCE_TARGET osx-arm64)
set(ONNX_INFERENCE_VERSION "1.17.3" CACHE STRING "ONNXRuntime version" FORCE)

if (ORT_ONLINE_COMPILED)
    set(ONNX_INFERENCE_PATH ${ONNX_PATH}/${ONNX_INFERENCE_TARGET}-${ONNX_INFERENCE_VERSION})
    message("[onnx.runtime.sd][I] Checking local onnxruntime path: ${ONNX_INFERENCE_PATH}")
    # download from github if ONNXRuntime library is not exists
    if (NOT EXISTS ${ONNX_INFERENCE_PATH})
        set(ONNXRuntime_Filename "onnxruntime-${ONNX_INFERENCE_TARGET}-${ONNX_INFERENCE_VERSION}.tgz")
        set(ONNXRuntime_Download https://github.com/microsoft/onnxruntime/releases/download/${${ONNX_INFERENCE_VERSION}}/${ONNXRuntime_Filename})
        message("[onnx.runtime.sd][I] Downloading onnxruntime library: ${ONNXRuntime_Download}")
        download_and_decompress(${ONNXRuntime_Download} ${ONNXRuntime_Filename} ${ONNX_INFERENCE_PATH})
    else ()
        message("[onnx.runtime.sd][I] Found local onnxruntime library: ${ONNX_INFERENCE_PATH}")
    endif ()

    if (NOT EXISTS ${ONNX_INFERENCE_PATH})
        message(FATAL_ERROR "[onnx.runtime.sd][E] ${ONNX_INFERENCE_PATH} is not exists!")
    endif ()

    include_directories(${ONNX_INFERENCE_PATH}/include)
    link_directories(${ONNX_INFERENCE_PATH}/lib)
else ()
    set(ONNX_INFERENCE_PATH ${ONNX_PATH}/onnxruntime)
    message("[onnx.runtime.sd][I] Checking local onnxruntime path: ${ONNX_INFERENCE_PATH}")
    add_subdirectory(${ONNX_INFERENCE_PATH})
    # prepare ort local compiling setting
    add_definitions(-DEXECUTION_PROVIDER_TENSORRT)
endif ()


#
# 3. Assemble project resource
#
set(ORT_SD_LIB ort-sd)

file(GLOB ORT_LIB_SOURCES
        "*.h"
        "*.cpp"
        "*.hpp"
        )

# we can get only one share lib
if(ORT_BUILD_SHARED_LIBS)
    message("Build shared library")
    message(${ORT_LIB_SOURCES})
    set(BUILD_SHARED_LIBS OFF)
    add_library(${ORT_SD_LIB} SHARED ${ORT_LIB_SOURCES})
    target_compile_definitions(${ORT_SD_LIB} PRIVATE -DSD_BUILD_DLL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    message("Build static library")
    set(BUILD_SHARED_LIBS OFF)
    add_library(${ORT_SD_LIB} STATIC ${ORT_LIB_SOURCES})
endif()

# deps
target_link_libraries(${ORT_SD_LIB} PUBLIC onnxruntime)
target_compile_features(${ORT_SD_LIB} PUBLIC cxx_std_11)

# check example available
if (SD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

