cmake_minimum_required(VERSION 3.12)

# do prepare CMake policies & state
set(CMAKE_CXX_STANDARD 17)
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()
if(POLICY CMP0145)
    cmake_policy(SET CMP0145 NEW)
endif()
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()
if(POLICY CMP0079)
    cmake_policy(SET CMP0079 NEW)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)       # generate compile_commands.json information
set(CMAKE_POSITION_INDEPENDENT_CODE ON)     # Ensure all targets are compiled with -fPIC
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# do prepare project header
include(${CMAKE_CURRENT_SOURCE_DIR}/apex/project_cmake_colors.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/apex/project_cmake_static.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/apex/project_cmake_ortenv.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/apex/project_cmake_utils.cmake)

PROJECT("onnxruntime-sd")
SET(ONNX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/engine)
SET(ONNX_INFERENCE_TARGET undefined)
SET(ONNX_INFERENCE_VERSION "1.18.0" CACHE STRING "ONNXRuntime version" FORCE)

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

# Warning: Cuda & TensorRT can only available in local-build mode
#          Caused by online-official doesn't provide target provider.h
if (WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4)      # for Windows x32
    SET(ONNX_INFERENCE_TARGET win-x86)
    set(DEFAULT_CUDA_STATE OFF)     # can be ON when local-build
    set(DEFAULT_TRT_STATE OFF)
    set(DEFAULT_COREML_STATE OFF)
    set(DEFAULT_NNAPI_STATE OFF)
    set(CURRENT_PLATFORM win-x86)
    set(SD_ORT_ONLINE_AVAIL ON)
    set(ENGINE_NAME "ORT-CUDA-TensorRT")
elseif (WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)  # for Windows x64
    SET(ONNX_INFERENCE_TARGET win-x64-gpu-cuda12)
    set(DEFAULT_CUDA_STATE OFF)     # can be ON when local-build
    set(DEFAULT_TRT_STATE OFF)      # can be ON when local-build
    set(DEFAULT_COREML_STATE OFF)
    set(DEFAULT_NNAPI_STATE OFF)
    set(CURRENT_PLATFORM win-x64)
    set(SD_ORT_ONLINE_AVAIL ON)
    set(ENGINE_NAME "ORT-CUDA-TensorRT")
elseif (APPLE)                      # for MacOS X or iOS, watchOS, tvOS (since 3.10.3)
    SET(ONNX_INFERENCE_TARGET osx-${CMAKE_SYSTEM_PROCESSOR})
    set(DEFAULT_CUDA_STATE OFF)
    set(DEFAULT_TRT_STATE OFF)
    set(DEFAULT_COREML_STATE ON)
    set(DEFAULT_NNAPI_STATE OFF)
    get_apple_platform(CURRENT_PLATFORM)
    set(SD_ORT_ONLINE_AVAIL ON)
    set(ENGINE_NAME "ORT-CoreML")
elseif (LINUX)                      # for Linux, BSD, Solaris, Minix
    SET(ONNX_INFERENCE_TARGET linux-${CMAKE_SYSTEM_PROCESSOR}-gpu-cuda12)
    set(DEFAULT_CUDA_STATE OFF)     # can be ON when local-build
    set(DEFAULT_TRT_STATE OFF)
    set(DEFAULT_COREML_STATE OFF)
    set(DEFAULT_NNAPI_STATE OFF)
    set(CURRENT_PLATFORM Linux)
    set(SD_ORT_ONLINE_AVAIL ON)
    set(ENGINE_NAME "ORT-CUDA")
elseif (ANDROID)                    # for Android
    SET(ONNX_INFERENCE_TARGET android)  # android
    set(DEFAULT_CUDA_STATE OFF)
    set(DEFAULT_TRT_STATE OFF)
    set(DEFAULT_COREML_STATE OFF)
    set(DEFAULT_NNAPI_STATE ON)
    set(CURRENT_PLATFORM Android)
    set(SD_ORT_ONLINE_AVAIL ON)
    set(SD_STANDALONE OFF)

    # force close Online-compiled for Android currently dont have such things
#    set(ORT_COMPILED_ONLINE OFF)
    set(ORT_BUILD_COMMAND_LINE OFF)
    set(ENGINE_NAME "ORT-CPU-NNAPI")
    message("[onnx.runtime.sd][I] target arch abi set to ${Red}${CMAKE_SYSTEM_PROCESSOR}:${CMAKE_ANDROID_ARCH_ABI}${ColourReset} minsdk ${Cyan}${CMAKE_SYSTEM_VERSION}${ColourReset}")
endif ()
message("[onnx.runtime.sd][I] target platform set to ${Red}${CMAKE_SYSTEM_NAME}${ColourReset}")
message("[onnx.runtime.sd][I] target building set to ${Red}${CMAKE_BUILD_TYPE}${ColourReset}")


message("${Cyan}<############################ ${ENGINE_NAME}-Config ############################>${ColourReset}")
# 1. Option list
option(ORT_COMPILED_ONLINE           "ort-sd: using online onnxruntime(ort), otherwise local build" ${SD_ORT_ONLINE_AVAIL})
option(ORT_COMPILED_HEAVY            "ort-sd: using HEAVY compile, ${Red}only for debug, default OFF${ColourReset}" OFF)
option(ORT_BUILD_COMMAND_LINE        "ort-sd: build command line tools" ${SD_STANDALONE})
option(ORT_BUILD_COMBINE_BASE        "ort-sd: build combine code together to build a single output lib" OFF)
option(ORT_BUILD_SHARED_CFDI         "ort-sd: build CFDI project shared libs" OFF)
option(ORT_BUILD_SHARED_LIBS         "ort-sd: build ORT in shared libs" OFF)
option(ORT_ENABLE_TENSOR_RT          "ort-sd: using TensorRT provider to accelerate inference" ${DEFAULT_TRT_STATE})
option(ORT_ENABLE_CUDA               "ort-sd: using CUDA provider to accelerate inference" ${DEFAULT_CUDA_STATE})
option(ORT_ENABLE_COREML             "ort-sd: using CoreML provider to accelerate inference" ${DEFAULT_COREML_STATE})
option(ORT_ENABLE_NNAPI              "ort-sd: using NNAPI provider to accelerate inference" ${DEFAULT_NNAPI_STATE})

if (ORT_BUILD_COMBINE_BASE)
    set(ORT_BUILD_SHARED_LIBS OFF)  # in this mode, we will never use ORT_BUILD_SHARED_LIBS option
    set(ORT_COMPILED_ONLINE OFF)
endif ()
if (ORT_COMPILED_ONLINE)            # ORT-static can only been built by local
    message("[onnx.runtime.sd][I] onnxruntime ${Blue}ORT_COMPILED_ONLINE:OFF${ColourReset} ${Red}enable ort.shared lib${ColourReset} ")
    set(ORT_BUILD_SHARED_LIBS ON)
endif ()
if (NOT ORT_BUILD_SHARED_LIBS AND NOT ARCHIVER_PATH)    # ORT-static can only been built with ar-tool helping
    message("[onnx.runtime.sd][I] onnxruntime ${Blue}ARCHIVER_PATH:NaN (Unset)${ColourReset} ${Red}disable ort.static lib${ColourReset} ")
    set(ORT_BUILD_SHARED_LIBS ON)
endif ()

set(option_state "option_state {\n")
set(option_state "${option_state}    compile  {\n")
set(option_state "${option_state}        ORT_COMPILED_ONLINE   : ${Cyan}${ORT_COMPILED_ONLINE}${ColourReset},\n")
set(option_state "${option_state}        ORT_COMPILED_HEAVY    : ${Red}${ORT_COMPILED_HEAVY}${ColourReset} ([W] Only for Debug using),\n")
set(option_state "${option_state}    }\n")
set(option_state "${option_state}    building {\n")
set(option_state "${option_state}        ORT_BUILD_COMMAND_LINE: ${Cyan}${ORT_BUILD_COMMAND_LINE}${ColourReset},\n")
set(option_state "${option_state}        ORT_BUILD_COMBINE_BASE: ${Cyan}${ORT_BUILD_COMBINE_BASE}${ColourReset},\n")
set(option_state "${option_state}        ORT_BUILD_SHARED_CFDI : ${Cyan}${ORT_BUILD_SHARED_CFDI}${ColourReset},\n")
set(option_state "${option_state}        ORT_BUILD_SHARED_LIBS : ${Cyan}${ORT_BUILD_SHARED_LIBS}${ColourReset},\n")
set(option_state "${option_state}    }\n")
set(option_state "${option_state}    provider {\n")
set(option_state "${option_state}        ORT_ENABLE_TENSOR_RT  : ${Cyan}${ORT_ENABLE_TENSOR_RT}${ColourReset},\n")
set(option_state "${option_state}        ORT_ENABLE_CUDA       : ${Cyan}${ORT_ENABLE_CUDA}${ColourReset},\n")
set(option_state "${option_state}        ORT_ENABLE_COREML     : ${Cyan}${ORT_ENABLE_COREML}${ColourReset},\n")
set(option_state "${option_state}        ORT_ENABLE_NNAPI      : ${Cyan}${ORT_ENABLE_NNAPI}${ColourReset},\n")
set(option_state "${option_state}    }\n")
set(option_state "${option_state}}")
message("${option_state}")

message("${Cyan}<############################# ${ENGINE_NAME}-Make #############################>${ColourReset}")

# 2. Add inference engine to project
auto_check_reference_submodule()
#auto_include(${ONNX_PATH}/onnxruntime include) DO NOT USE onnxruntime/include for it's not usable

# 3. Check Provider Enable State
if (ORT_ENABLE_TENSOR_RT)
    message("[onnx.runtime.sd][I] onnxruntime enable ${Red}TensorRT Provider${ColourReset}")
    add_definitions(-DENABLE_TENSOR_RT)
endif ()
if (ORT_ENABLE_CUDA)
    message("[onnx.runtime.sd][I] onnxruntime enable ${Red}CUDA Provider${ColourReset}")
    add_definitions(-DENABLE_CUDA)
endif ()
if (ORT_ENABLE_COREML)
    message("[onnx.runtime.sd][I] onnxruntime enable ${Red}CoreML Provider${ColourReset}")
    add_definitions(-DENABLE_COREML)
endif ()
if (ORT_ENABLE_NNAPI)
    message("[onnx.runtime.sd][I] onnxruntime enable ${Red}NNAPI Provider${ColourReset}")
    add_definitions(-DENABLE_NNAPI)
endif ()
message("[onnx.runtime.sd][I] onnxruntime enable ${Red}CPU Provider${ColourReset}")

# 4. Prepare inference engine compiled library (from online/local)
if (ORT_BUILD_COMBINE_BASE)
    # Enter: combine ort-lib
    set(onnxruntime_USE_TENSORRT ${ORT_ENABLE_TENSOR_RT})
    set(onnxruntime_TENSORRT_HOME ${NVIDIA_TENSORRT_PATH})
    set(onnxruntime_USE_CUDA ${ORT_ENABLE_CUDA})
    set(onnxruntime_CUDNN_HOME ${NVIDIA_CUDNN_PATH})
    set(onnxruntime_CUDA_HOME ${NVIDIA_CUDA_PATH})
    set(onnxruntime_USE_COREML ${ORT_ENABLE_COREML})
    set(onnxruntime_USE_NNAPI_BUILTIN ${ORT_ENABLE_NNAPI})
    add_subdirectory(engine/onnxruntime/cmake ${CMAKE_BINARY_DIR}/onnxruntime)
elseif (ORT_COMPILED_ONLINE)
    # Enter: from online/local_path
    message("[onnx.runtime.sd][I] ${Blue}onnxruntime build mode:${ColourReset} ${Red}Online${ColourReset} start")
    set(ONNX_INFERENCE_PATH ${ONNX_PATH}/onnxruntime-${ONNX_INFERENCE_TARGET}-${ONNX_INFERENCE_VERSION})
    message("[onnx.runtime.sd][I] Checking local onnxruntime path: ${ONNX_INFERENCE_PATH}")

    if (WIN32)
        set(ONNXRuntime_Filename "onnxruntime-${ONNX_INFERENCE_TARGET}-${ONNX_INFERENCE_VERSION}.zip")
        set(ONNXRuntime_Download "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_INFERENCE_VERSION}/${ONNXRuntime_Filename}")
    elseif (ANDROID)
        set(ONNXRuntime_Filename "onnxruntime-${ONNX_INFERENCE_TARGET}-${ONNX_INFERENCE_VERSION}.aar")
        set(ONNXRuntime_Download "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ONNX_INFERENCE_VERSION}/${ONNXRuntime_Filename}")
    else ()
        set(ONNXRuntime_Filename "onnxruntime-${ONNX_INFERENCE_TARGET}-${ONNX_INFERENCE_VERSION}.tgz")
        set(ONNXRuntime_Download "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_INFERENCE_VERSION}/${ONNXRuntime_Filename}")
    endif ()

    if (NOT EXISTS ${ONNX_INFERENCE_PATH})
        message("[onnx.runtime.sd][I] Downloading onnxruntime library: ${ONNXRuntime_Download}")
        download_and_decompress(${ONNXRuntime_Download} ${ONNXRuntime_Filename} ${ONNX_INFERENCE_PATH})
    else ()
        message("[onnx.runtime.sd][I] Found local onnxruntime library: ${ONNX_INFERENCE_PATH}")
    endif ()

    if (NOT EXISTS ${ONNX_INFERENCE_PATH})
        message(FATAL_ERROR "[onnx.runtime.sd][E] ${ONNX_INFERENCE_PATH} does not exist!")
    endif ()

    if (ANDROID)
        set(ONNXRUNTIME_LIB_PATH ${ONNX_INFERENCE_PATH}/jni/${CMAKE_ANDROID_ARCH_ABI})
    else ()
        set(ONNXRUNTIME_LIB_PATH ${ONNX_INFERENCE_PATH}/lib)
    endif ()
    link_directories(${ONNXRUNTIME_LIB_PATH})
    message("[onnx.runtime.sd][I] ${Blue}onnxruntime build mode:${ColourReset} ${Red}Online${ColourReset} done")
else ()
    # Enter: local built
    message("[onnx.runtime.sd][I] ${Blue}onnxruntime build mode:${ColourReset} ${Red}Local${ColourReset} start")
    auto_switch_ort_build_type()
    set(ONNX_INFERENCE_PATH ${ONNX_PATH}/onnxruntime/build/${CURRENT_PLATFORM}/${ORT_BUILD_TYPE})
    message("[onnx.runtime.sd][I] Checking local onnxruntime path: ${ONNX_INFERENCE_PATH}")
    # download from github if ONNXRuntime library is not exists
    check_library_exists(onnxruntime ${ONNX_INFERENCE_PATH} LIBRARY_FOUND)
    if (NOT LIBRARY_FOUND)
        auto_build_reference_submodule()
        auto_merge_submodule_compiled()
    else ()
        message("[onnx.runtime.sd][I] Found local onnxruntime library: ${ONNX_INFERENCE_PATH}/libonnxruntime.so|a")
    endif ()

    check_library_exists(onnxruntime ${ONNX_INFERENCE_PATH} LIBRARY_FOUND)
    if (NOT LIBRARY_FOUND)
        message(FATAL_ERROR "[onnx.runtime.sd][E] ${ONNX_INFERENCE_PATH}/libonnxruntime.so|a not exists!")
    endif ()

    set(ONNXRUNTIME_LIB_PATH ${ONNX_INFERENCE_PATH})
    link_directories(${ONNXRUNTIME_LIB_PATH})
    message("[onnx.runtime.sd][I] ${Blue}onnxruntime build mode:${ColourReset} ${Red}Local${ColourReset} done")
endif ()

message("[onnx.runtime.sd][I] onnxruntime target ${Blue}version${ColourReset}: ${ONNX_INFERENCE_VERSION}")
message("[onnx.runtime.sd][I] onnxruntime target ${Blue}platform${ColourReset}: ${ONNX_INFERENCE_TARGET}")
message("[onnx.runtime.sd][I] onnxruntime infers ${Blue}provider${ColourReset}: ${ENGINE_NAME}")
message("${Cyan}<############################# ${ENGINE_NAME}-Done #############################>${ColourReset}")


message("${Cyan}<############################# ${PROJECT_NAME}-Make #############################>${ColourReset}")

set(library_name ort-sd)

set(library_data_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/outlet/ort_sd.cc
)

if (ORT_COMPILED_HEAVY)
    auto_target_sources(library_data_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} source)
    auto_target_sources(library_data_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} outlet)
endif ()
auto_target_sources(library_data_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} include)
auto_print_list(library_data_SOURCES)

# we can get only one share lib
if(ORT_BUILD_SHARED_CFDI)
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
auto_link_reference_library(${library_name} onnxruntime ${ONNXRUNTIME_LIB_PATH})

target_compile_definitions(${library_name} PUBLIC ${CMAKE_BUILD_TYPE})

message("${Cyan}<############################# ${PROJECT_NAME}-Done #############################>${ColourReset}")


# check command line available
if (ORT_BUILD_COMMAND_LINE)
    message("[onnx.runtime.sd][I] build command line tools at ${Red}${CMAKE_SYSTEM_NAME}${ColourReset}")
    add_subdirectory(clitools)
endif()

