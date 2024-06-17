cmake_minimum_required(VERSION 3.12)

include(${CMAKE_CURRENT_SOURCE_DIR}/apex/project_cmake_colors.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/apex/project_cmake_ortenv.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/apex/project_cmake_utils.cmake)

PROJECT("onnxruntime-sd")
SET(ENGINE_NAME "ORT-TensorRT")
SET(ONNX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/engine)
SET(ONNX_INFERENCE_TARGET osx-arm64)
SET(ONNX_INFERENCE_VERSION "1.18.0" CACHE STRING "ONNXRuntime version" FORCE)

# Ensure all targets are compiled with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# do prepare CMake state
if(POLICY CMP0145)
    cmake_policy(SET CMP0145 OLD)
endif()
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 OLD)
endif()
if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# do prepare option auto enable, based on paltform
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SD_STANDALONE ON)
else()
    set(SD_STANDALONE OFF)
endif()

if (WIN32)                          # for Windows x32
    SET(ONNX_INFERENCE_TARGET win-x86)
    set(DEFAULT_CUDA_STATE ON)
    set(DEFAULT_TRT_STATE OFF)
    set(DEFAULT_COREML_STATE OFF)
    set(CURRENT_PLATFORM win-x86)
    set(SD_ORT_ONLINE_AVAIL ON)
elseif (WIN64)                      # for Windows x64
    SET(ONNX_INFERENCE_TARGET win-x64-gpu-cuda12)
    set(DEFAULT_CUDA_STATE ON)
    set(DEFAULT_TRT_STATE ON)
    set(DEFAULT_COREML_STATE OFF)
    set(CURRENT_PLATFORM win-x64)
    set(SD_ORT_ONLINE_AVAIL ON)
elseif (APPLE)                      # for MacOS X or iOS, watchOS, tvOS (since 3.10.3)
    SET(ONNX_INFERENCE_TARGET osx-${CMAKE_SYSTEM_PROCESSOR})
    set(DEFAULT_CUDA_STATE OFF)
    set(DEFAULT_TRT_STATE OFF)
    set(DEFAULT_COREML_STATE ON)
    set(CURRENT_PLATFORM Apple)
    set(SD_ORT_ONLINE_AVAIL ON)
elseif (LINUX)                      # for Linux, BSD, Solaris, Minix
    SET(ONNX_INFERENCE_TARGET linux-${CMAKE_SYSTEM_PROCESSOR}-gpu-cuda12)
    set(DEFAULT_CUDA_STATE ON)
    set(DEFAULT_TRT_STATE OFF)
    set(DEFAULT_COREML_STATE OFF)
    set(CURRENT_PLATFORM Linux)
    set(SD_ORT_ONLINE_AVAIL ON)
elseif (ANDROID)                    # for Android
    SET(ONNX_INFERENCE_TARGET android-${CMAKE_SYSTEM_PROCESSOR})  # android-arm64-v8a
    set(DEFAULT_CUDA_STATE OFF)
    set(DEFAULT_TRT_STATE OFF)
    set(DEFAULT_COREML_STATE OFF)
    set(CURRENT_PLATFORM Android)
    set(SD_ORT_ONLINE_AVAIL ON)
    set(SD_STANDALONE OFF)

    # force close Online-compiled for Android currently dont have such things
    set(ORT_ONLINE_COMPILED OFF)
    set(ORT_LOCALS_BUILDING ON)
    set(ORT_BUILD_COMMAND_LINE OFF)
endif ()
message("[onnx.runtime.sd][I] target platform set to ${Red}${CMAKE_SYSTEM_NAME}${ColourReset}")

message("${Cyan}<############################# ${ENGINE_NAME}-Make #############################>${ColourReset}")
# 1. Option list
option(ORT_BUILD_COMMAND_LINE        "ort-sd: build command line tools" ${SD_STANDALONE})
option(ORT_BUILD_SHARED_LIBS         "ort-sd: build shared libs" OFF)
option(ORT_ONLINE_COMPILED           "ort-sd: using online onnxruntime(ort)" ${SD_ORT_ONLINE_AVAIL})
option(ORT_LOCALS_BUILDING           "ort-sd: using self-build onnxruntime(ort)" OFF)
option(ORT_ENABLE_TENSOR_RT          "ort-sd: using TensorRT provider to accelerate inference" ${DEFAULT_TRT_STATE})
option(ORT_ENABLE_CUDA               "ort-sd: using CUDA provider to accelerate inference" ${DEFAULT_CUDA_STATE})
option(ORT_ENABLE_COREML             "ort-sd: using CoreML provider to accelerate inference" ${DEFAULT_COREML_STATE})

# 2. Add inference engine to project
auto_check_reference_submodule()
auto_include(${ONNX_PATH}/onnxruntime include)

# 3. Prepare inference engine compiled library (from online/local)
if (ORT_ONLINE_COMPILED)
    set(ONNX_INFERENCE_PATH ${ONNX_PATH}/onnxruntime-${ONNX_INFERENCE_TARGET}-${ONNX_INFERENCE_VERSION})
    message("[onnx.runtime.sd][I] Checking local onnxruntime path: ${ONNX_INFERENCE_PATH}")
    # download from github if ONNXRuntime library is not exists
    if (NOT EXISTS ${ONNX_INFERENCE_PATH})
        set(ONNXRuntime_Filename "onnxruntime-${ONNX_INFERENCE_TARGET}-${ONNX_INFERENCE_VERSION}.tgz")
        set(ONNXRuntime_Download https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_INFERENCE_VERSION}/${ONNXRuntime_Filename})
        message("[onnx.runtime.sd][I] Downloading onnxruntime library: ${ONNXRuntime_Download}")
        download_and_decompress(${ONNXRuntime_Download} ${ONNXRuntime_Filename} ${ONNX_INFERENCE_PATH})
    else ()
        message("[onnx.runtime.sd][I] Found local onnxruntime library: ${ONNX_INFERENCE_PATH}")
    endif ()

    if (NOT EXISTS ${ONNX_INFERENCE_PATH})
        message(FATAL_ERROR "[onnx.runtime.sd][E] ${ONNX_INFERENCE_PATH} is not exists!")
    endif ()

    link_directories(${ONNX_INFERENCE_PATH}/lib)
endif ()

if (ORT_LOCALS_BUILDING)
    auto_switch_ort_build_type()
    set(ONNX_INFERENCE_PATH ${ONNX_PATH}/onnxruntime/build/${CURRENT_PLATFORM}/${ORT_BUILD_TYPE})
    message("[onnx.runtime.sd][I] Checking local onnxruntime path: ${ONNX_INFERENCE_PATH}")
    # download from github if ONNXRuntime library is not exists
    if (NOT EXISTS ${ONNX_INFERENCE_PATH}/libonnxruntime.a)
        # do check ORT ./engine/onnxruntime exist
        auto_build_reference_submodule()
        auto_merge_submodule_compiled()
    else ()
        message("[onnx.runtime.sd][I] Found local onnxruntime library: ${ONNX_INFERENCE_PATH}/libonnxruntime.a")
    endif ()

    if (NOT EXISTS ${ONNX_INFERENCE_PATH}/libonnxruntime.a)
        message(FATAL_ERROR "[onnx.runtime.sd][E] ${ONNX_INFERENCE_PATH}/libonnxruntime.a not exists!")
    endif ()

    link_directories(${ONNX_INFERENCE_PATH}/lib)
endif ()

# 4. Check Provider Enable State
if (ORT_ENABLE_TENSOR_RT)
    message("[onnx.runtime.sd][I] onnxruntime enable ${Red}TensorRT Provider${ColourReset}")
    add_definitions(-DENABLE_TENSOR_RT)
endif ()
if (ORT_ENABLE_CUDA)
    message("[onnx.runtime.sd][I] onnxruntime enable ${Red}CUDA Provider${ColourReset}")
    add_definitions(-DENABLE_CUDA)
endif ()
if (ORT_ENABLE_COREML)
    message("[onnx.runtime.sd][I] onnxruntime enable ${Red}TensorRT Provider${ColourReset}")
    add_definitions(-DENABLE_COREML)
endif ()
message("[onnx.runtime.sd][I] onnxruntime enable ${Red}CPU Provider${ColourReset}")

message("[onnx.runtime.sd][I] onnxruntime target ${Blue}version${ColourReset}: ${ONNX_INFERENCE_VERSION}")
message("[onnx.runtime.sd][I] onnxruntime target ${Blue}platform${ColourReset}: ${ONNX_INFERENCE_TARGET}")
message("[onnx.runtime.sd][I] onnxruntime infers ${Blue}provider${ColourReset}: ${ENGINE_NAME}")
message("${Cyan}<############################# ${ENGINE_NAME}-Done #############################>${ColourReset}")


message("${Cyan}<############################# ${PROJECT_NAME}-Make #############################>${ColourReset}")

set(library_name ort-sd)

set(library_data_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/outlet/ort_sd.cc
)

auto_target_sources(library_data_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} source)
auto_target_sources(library_data_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} outlet)
auto_target_sources(library_data_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} include)
auto_print_list(library_data_SOURCES)

# we can get only one share lib
if(ORT_BUILD_SHARED_LIBS)
    message("[onnx.runtime.sd][I] Build library[SHARED]: ${library_name}")
    add_library(${library_name} SHARED)
    target_compile_definitions(${library_name} PRIVATE -DSD_BUILD_DLL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    message("[onnx.runtime.sd][I] ${Blue}Build library[STATIC]${ColourReset}: ${library_name}")
    add_library(${library_name} STATIC)
    set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
endif()

target_sources(${library_name} PRIVATE ${library_data_SOURCES})
auto_target_include(${library_name} ${CMAKE_CURRENT_SOURCE_DIR} source PRIVATE)
auto_target_include(${library_name} ${CMAKE_CURRENT_SOURCE_DIR} outlet PRIVATE)
auto_target_include(${library_name} ${CMAKE_CURRENT_SOURCE_DIR} include PUBLIC)

#link references
auto_link_reference_library(${library_name})

target_compile_features(${library_name} PUBLIC cxx_std_14)
target_compile_definitions(${library_name} PUBLIC ${CMAKE_BUILD_TYPE})

message("${Cyan}<############################# ${PROJECT_NAME}-Done #############################>${ColourReset}")


# check command line available
if (ORT_BUILD_COMMAND_LINE)
    message("[onnx.runtime.sd][I] build command line tools at ${Red}${CMAKE_SYSTEM_NAME}${ColourReset}")
    add_subdirectory(clitools)
endif()

