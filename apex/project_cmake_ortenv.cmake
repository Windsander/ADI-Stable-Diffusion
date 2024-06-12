# Defines functions and macros useful for ORT CMakeLists using.
# Created:
#
# - by Arikan.Li on 2024/06/01.
#
# Note:
#
# - ???What?

# 平台限定=================================================================================================
#自动检测关联项目 onnxruntime 的编译版本
macro(auto_switch_ort_build_type)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(ORT_BUILD_TYPE Debug)
        message(STATUS "[onnx.runtime.sd][I] set ORT to Debug build")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(ORT_BUILD_TYPE Release)
        message(STATUS "[onnx.runtime.sd][I] set ORT to Release build")
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        set(ORT_BUILD_TYPE RelWithDebInfo)
        message(STATUS "[onnx.runtime.sd][I] set ORT to Release with Debug Info build")
    elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        set(ORT_BUILD_TYPE MinSizeRel)
        message(STATUS "[onnx.runtime.sd][I] set ORT to Minimum Size Release build")
    else()
        set(ORT_BUILD_TYPE MinSizeRel)
        message(STATUS "[onnx.runtime.sd][I] Unknown build type, set ORT to default(MinSizeRel)")
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
        # do prepare option auto enable, based on paltform
        if (WIN32)                          # for Windows x32
            execute_process(
                    COMMAND ${ONNX_PATH}/onnxruntime/build.bat
                            --config ${ORT_BUILD_TYPE}
                            --parallel
                            --compile_no_warning_as_error
                    WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
                    RESULT_VARIABLE ${ONNX_PATH}/result
            )
            if (result)
                message(FATAL_ERROR "[onnx.runtime.sd][E] Failed to build WIN32 ONNXRuntime")
            endif ()
        elseif (WIN64)                      # for Windows x64
            execute_process(
                    COMMAND ${ONNX_PATH}/onnxruntime/build.bat
                            -–arm64
                            --config ${ORT_BUILD_TYPE}
                            --parallel
                            --compile_no_warning_as_error
                    WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
                    RESULT_VARIABLE ${ONNX_PATH}/result
            )
            if (result)
                message(FATAL_ERROR "[onnx.runtime.sd][E] Failed to build WIN64 ONNXRuntime")
            endif ()
        elseif (APPLE)                      # for MacOS X or iOS, watchOS, tvOS (since 3.10.3)
            execute_process(
                    COMMAND ${ONNX_PATH}/onnxruntime/build.sh
                            --config ${ORT_BUILD_TYPE}
                            --parallel
                            --compile_no_warning_as_error
                            --cmake_extra_defines CMAKE_OSX_ARCHITECTURES="x86_64;arm64"
                    WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
                    RESULT_VARIABLE ${ONNX_PATH}/result
            )
            if (result)
                message(FATAL_ERROR "[onnx.runtime.sd][E] Failed to build iOS/OSX ONNXRuntime")
            endif ()
        elseif (LINUX)                      # for Linux, BSD, Solaris, Minix
            execute_process(
                    COMMAND ${ONNX_PATH}/onnxruntime/build.sh
                            --config ${ORT_BUILD_TYPE}
                            --parallel
                            --compile_no_warning_as_error
                    WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
                    RESULT_VARIABLE ${ONNX_PATH}/result
            )
            if (result)
                message(FATAL_ERROR "[onnx.runtime.sd][E] Failed to build Linux ONNXRuntime")
            endif ()
        elseif (ANDROID)                    # for Android
            execute_process(
                    COMMAND ${ONNX_PATH}/onnxruntime/build.sh
                            --android
                            --config ${ORT_BUILD_TYPE}
                            --android_sdk_path ${ANDROID_SDK}
                            --android_ndk_path ${ANDROID_NDK}
                            --android_abi ${CMAKE_ANDROID_ARCH_ABI}
                            --android_api ${CMAKE_SYSTEM_VERSION}
                            --use_nnapi
                            --parallel
                            --skip_tests
                    WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
                    RESULT_VARIABLE ${ONNX_PATH}/result
            )
            if (result)
                message(FATAL_ERROR "[onnx.runtime.sd][E] Failed to build Android ONNXRuntime")
            endif ()
        endif ()
    else ()
        message(STATUS "[onnx.runtime.sd][I] Unfounded onnxruntime submodule. please clone from https://github.com/microsoft/onnxruntime.git first!")
    endif ()
endmacro()
