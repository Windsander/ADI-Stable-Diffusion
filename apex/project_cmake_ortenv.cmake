# Defines functions and macros useful for ORT CMakeLists using.
# Created:
#
# - by Arikan.Li on 2024/06/01.
#
# Note:
#
# - ???What?

# 平台限定=================================================================================================
#自动检测当前 Apple 平台类型
function(get_apple_platform result_var)
    if(APPLE)
        if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
            if(CMAKE_OSX_SYSROOT MATCHES ".*iPhone.*")
                set(platform "iOS")
            elseif(CMAKE_OSX_SYSROOT MATCHES ".*AppleTV.*")
                set(platform "tvOS")
            elseif(CMAKE_OSX_SYSROOT MATCHES ".*Watch.*")
                set(platform "watchOS")
            else()
                set(platform "MacOS")
            endif()
        else()
            set(platform "UnknownApplePlatform")
        endif()
    else()
        set(platform "NotApplePlatform")
    endif()
    set(${result_var} "${platform}" PARENT_SCOPE)
endfunction()

#自动检测关联项目 onnxruntime 的编译版本
macro(auto_switch_ort_build_type)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(ORT_BUILD_TYPE Debug)
        message(STATUS "[onnx.runtime.sd][I] set ${Blue}ORT${ColourReset} to ${Red}Debug${ColourReset} build")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(ORT_BUILD_TYPE Release)
        message(STATUS "[onnx.runtime.sd][I] set ${Blue}ORT${ColourReset} to ${Red}Release${ColourReset} build")
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(ORT_BUILD_TYPE RelWithDebInfo)
        message(STATUS "[onnx.runtime.sd][I] set ${Blue}ORT${ColourReset} to ${Red}Release with Debug Info${ColourReset} build")
    elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        set(ORT_BUILD_TYPE MinSizeRel)
        message(STATUS "[onnx.runtime.sd][I] set ${Blue}ORT${ColourReset} to ${Red}Minimum Size Release${ColourReset} build")
    else()
        set(ORT_BUILD_TYPE MinSizeRel)
        message(STATUS "[onnx.runtime.sd][I] Unknown build type, set ${Blue}ORT${ColourReset} to ${Red}default(MinSizeRel)${ColourReset}")
    endif()
endmacro()

#自动检测关联项目 onnxruntime 是否以在本地
macro(auto_check_reference_submodule)
    # 动态检测 ./engine/onnxruntime 是否存在
    if (NOT EXISTS ${ONNX_PATH}/onnxruntime)
        message(STATUS "[onnx.runtime.sd][I] onnxruntime submodule not found. Cloning from GitHub...")
        execute_process(
                COMMAND git clone --recursive https://github.com/microsoft/onnxruntime.git ${ONNX_PATH}/onnxruntime
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE result
        )
        if (result)
            message(FATAL_ERROR "[onnx.runtime.sd][E] Failed to clone onnxruntime repository")
        endif ()
    else ()
        message(STATUS "[onnx.runtime.sd][I] Found onnxruntime submodule.")
    endif ()
endmacro()

#自动检测关联项目 onnxruntime 是否已生成
macro(auto_build_reference_submodule)
    # 动态检测 ./engine/onnxruntime 是否存在
    if (EXISTS ${ONNX_PATH}/onnxruntime)
        set(COMMAND_LIST ${ONNX_PATH}/onnxruntime/build)
        # do prepare option auto enable, based on platform
        if (WIN32)                      # for Windows x32 & x64
            string(APPEND COMMAND_LIST .bat)
            list(APPEND COMMAND_LIST --config ${ORT_BUILD_TYPE})
            list(APPEND COMMAND_LIST --parallel)
            list(APPEND COMMAND_LIST --compile_no_warning_as_error)
            if (CMAKE_SIZEOF_VOID_P EQUAL 8)
                list(APPEND COMMAND_LIST --arm64)
            endif ()
            if (ORT_BUILD_SHARED_LIBS)
                list(APPEND COMMAND_LIST --build_shared_lib)
            endif ()
            if (ORT_ENABLE_CUDA)
                list(APPEND COMMAND_LIST --use_cuda)
                list(APPEND COMMAND_LIST --cudnn_home ${NVIDIA_CUDNN_PATH})
                list(APPEND COMMAND_LIST --cuda_home ${NVIDIA_CUDA_PATH})
            endif ()
            if (ORT_ENABLE_TENSOR_RT)
                list(APPEND COMMAND_LIST --use_tensorrt)
                list(APPEND COMMAND_LIST --tensorrt_home ${NVIDIA_TENSORRT_PATH})
            endif ()
        elseif (APPLE)                  # for MacOS X or iOS, watchOS, tvOS (since 3.10.3)
            string(APPEND COMMAND_LIST .sh)
            list(APPEND COMMAND_LIST --config ${ORT_BUILD_TYPE})
            list(APPEND COMMAND_LIST --parallel)
            list(APPEND COMMAND_LIST --compile_no_warning_as_error)
            list(APPEND COMMAND_LIST --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=${CMAKE_SYSTEM_PROCESSOR})
            if (ORT_BUILD_SHARED_LIBS)
                list(APPEND COMMAND_LIST --build_shared_lib)
            endif ()
            if (ORT_ENABLE_COREML)
                list(APPEND COMMAND_LIST --use_coreml)
            endif ()
        elseif (LINUX)                  # for Linux, BSD, Solaris, Minix
            string(APPEND COMMAND_LIST .sh)
            list(APPEND COMMAND_LIST --config ${ORT_BUILD_TYPE})
            list(APPEND COMMAND_LIST --parallel)
            list(APPEND COMMAND_LIST --compile_no_warning_as_error)
            if (ORT_BUILD_SHARED_LIBS)
                list(APPEND COMMAND_LIST --build_shared_lib)
            endif ()
            if (ORT_ENABLE_CUDA)
                list(APPEND COMMAND_LIST --use_cuda)
                list(APPEND COMMAND_LIST --cudnn_home ${NVIDIA_CUDNN_PATH})
                list(APPEND COMMAND_LIST --cuda_home ${NVIDIA_CUDA_PATH})
            endif ()
            if (ORT_ENABLE_TENSOR_RT)
                list(APPEND COMMAND_LIST --use_tensorrt)
                list(APPEND COMMAND_LIST --tensorrt_home ${NVIDIA_TENSORRT_PATH})
            endif ()
        elseif (ANDROID)                # for Android
            string(APPEND COMMAND_LIST .sh)
            list(APPEND COMMAND_LIST --config ${ORT_BUILD_TYPE})
            list(APPEND COMMAND_LIST --parallel)
            list(APPEND COMMAND_LIST --android)
            list(APPEND COMMAND_LIST --android_sdk_path ${ANDROID_SDK})
            list(APPEND COMMAND_LIST --android_ndk_path ${ANDROID_NDK})
            list(APPEND COMMAND_LIST --android_abi ${CMAKE_ANDROID_ARCH_ABI})
            list(APPEND COMMAND_LIST --android_api ${CMAKE_SYSTEM_VERSION})
            list(APPEND COMMAND_LIST --minimal_build extended)
            if (ORT_BUILD_SHARED_LIBS)
                list(APPEND COMMAND_LIST --android_cpp_shared)
                list(APPEND COMMAND_LIST --build_shared_lib)
            endif ()
            list(APPEND COMMAND_LIST --skip_tests)
            if (ORT_ENABLE_NNAPI)
                list(APPEND COMMAND_LIST --use_nnapi)
            endif ()
        endif ()
    else ()
        message(FATAL_ERROR "[onnx.runtime.sd][E] Unfounded onnxruntime submodule. please clone from https://github.com/microsoft/onnxruntime.git first!")
    endif ()
    list(APPEND COMMAND_LIST --update)
    list(APPEND COMMAND_LIST --build)

    # begin ORT build
    message(STATUS "[onnx.runtime.sd][I] execute_process: ${Cyan}${COMMAND_LIST}${ColourReset}")
    execute_process(
            COMMAND ${COMMAND_LIST}
            WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
            RESULT_VARIABLE result
    )
    message(STATUS "[onnx.runtime.sd][I] execute_process build ${CMAKE_SYSTEM_NAME} ONNXRuntime done")
    if (result)
        message(FATAL_ERROR "[onnx.runtime.sd][E] Failed to build ${CMAKE_SYSTEM_NAME} ONNXRuntime")
    endif ()
endmacro()

#自动检测关联项目 onnxruntime 是否已生成
macro(auto_merge_submodule_compiled)
    if (ORT_BUILD_SHARED_LIBS)
        message(STATUS "[onnx.runtime.sd][I] no need for ORT.a build: ${ARCHIVER_PATH}")
    elseif (EXISTS ${ONNX_PATH}/onnxruntime)
        if (WIN32)                      # for Windows x32 & x64
            merge_static_libs_windows()
        elseif (APPLE)                  # for MacOS X or iOS, watchOS, tvOS (since 3.10.3)
            merge_static_libs_macos()
        elseif (LINUX)                  # for Linux, BSD, Solaris, Minix
            merge_static_libs_linux()
        elseif (ANDROID)                # for Android
            merge_static_libs_android()
        else()
            message(FATAL_ERROR "[onnx.runtime.sd][E] Unsupported platform!")
        endif()
    else ()
        message(FATAL_ERROR "[onnx.runtime.sd][E] Unfounded onnxruntime submodule. please clone from https://github.com/microsoft/onnxruntime.git first!")
    endif ()
endmacro()
