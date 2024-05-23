cmake_minimum_required(VERSION 3.12)

include(${CMAKE_CURRENT_SOURCE_DIR}/apex/renderer_cmake_colors.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/apex/renderer_cmake_utils.cmake)

PROJECT("onnxruntime-sd")
SET(ENGINE_NAME "ORT-TensorRT")
SET(ONNX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/engine)
SET(ONNX_INFERENCE_TARGET osx-arm64)
SET(ONNX_INFERENCE_VERSION "1.17.3" CACHE STRING "ONNXRuntime version" FORCE)

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

# do prepare option auto enable, based on paltform
if (WIN32 OR MSVC)                  # for Windows
    set(DEFAULT_CUDA_STATE ON)
    set(DEFAULT_TRT_STATE ON)
elseif (APPLE)                      # for MacOS X or iOS, watchOS, tvOS (since 3.10.3)
    set(DEFAULT_CUDA_STATE OFF)
    set(DEFAULT_TRT_STATE OFF)
elseif (UNIX AND NOT APPLE)         # for Linux, BSD, Solaris, Minix
    set(DEFAULT_CUDA_STATE ON)
    set(DEFAULT_TRT_STATE OFF)
endif ()


message("${Cyan}<############################# ${ENGINE_NAME}-Make #############################>${ColourReset}")
# 1. Option list
option(ORT_BUILD_COMMAND_LINE        "ort-sd: build command line tools" ${SD_STANDALONE})
option(ORT_BUILD_SHARED_LIBS         "ort-sd: build shared libs" OFF)
option(ORT_ONLINE_COMPILED           "ort-sd: using online onnxruntime(ort)" ON)
option(ORT_LOCALS_BUILDING           "ort-sd: using self-build onnxruntime(ort)" OFF)
option(ORT_ENABLE_CUDA               "ort-sd: using CUDA provider to accelerate inference" ${DEFAULT_CUDA_STATE})
option(ORT_ENABLE_TENSOR_RT          "ort-sd: using TensorRT provider to accelerate inference" ${DEFAULT_TRT_STATE})

# 2. Add inference engine to project
if (ORT_ONLINE_COMPILED)
    set(ONNX_INFERENCE_PATH ${ONNX_PATH}/onnxruntime-${ONNX_INFERENCE_TARGET}-${ONNX_INFERENCE_VERSION})
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
endif ()

if (ORT_LOCALS_BUILDING)
    set(ONNX_INFERENCE_PATH ${ONNX_PATH}/onnxruntime)
    message("[onnx.runtime.sd][I] Checking local onnxruntime path: ${ONNX_INFERENCE_PATH}")
    add_subdirectory(${ONNX_INFERENCE_PATH})
    # prepare ort local compiling setting
    add_definitions(-DEXECUTION_PROVIDER_TENSORRT)
    message("[onnx.runtime.sd][W] ${Red}Using local self-build ort, u must define a ort cmake file by u own${ColourReset}")
endif ()

# 3. Check Provider Enable State
if (ORT_ENABLE_CUDA)
    message("[onnx.runtime.sd][I] onnxruntime enable ${Red}CUDA Provider${ColourReset}")
    add_definitions(-DENABLE_CUDA)
endif ()
if (ORT_ENABLE_TENSOR_RT)
    message("[onnx.runtime.sd][I] onnxruntime enable ${Red}TensorRT Provider${ColourReset}")
    add_definitions(-DENABLE_TENSOR_RT)
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
    add_subdirectory(clitools)
endif()

