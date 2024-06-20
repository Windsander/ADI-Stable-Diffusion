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
            list(APPEND COMMAND_LIST --cmake_extra_defines CMAKE_OSX_ARCHITECTURES="x86_64;arm64")
            if (ORT_ENABLE_COREML)
                list(APPEND COMMAND_LIST --use_coreml)
            endif ()
        elseif (LINUX)                  # for Linux, BSD, Solaris, Minix
            string(APPEND COMMAND_LIST .sh)
            list(APPEND COMMAND_LIST --config ${ORT_BUILD_TYPE})
            list(APPEND COMMAND_LIST --parallel)
            list(APPEND COMMAND_LIST --compile_no_warning_as_error)
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
            list(APPEND COMMAND_LIST --disable_exceptions)
            list(APPEND COMMAND_LIST --android_cpp_shared)
            list(APPEND COMMAND_LIST --build_shared_lib)
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
    # 动态检测 ./engine/onnxruntime 是否存在
    if (EXISTS ${ONNX_PATH}/onnxruntime)
        if(NOT DEFINED ONNX_INFERENCE_PATH OR ONNX_INFERENCE_PATH STREQUAL "")
            message(FATAL_ERROR "[onnx.runtime.sd][E] ONNX_INFERENCE_PATH is not set or is empty")
        endif()

        # 动态生成 merge.mri 文件
        string(
                CONCAT merge_mri_content
                "create libonnxruntime.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_common.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_framework.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_graph.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_mlas.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_optimizer.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_providers.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_session.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_util.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/external/onnx/libonnx.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/lib/libgmock.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/external/onnx/libonnx_proto.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/lib/libgtest.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/external/nsync/libnsync_cpp.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/external/re2/libre2.a\n"
                "addlib ${ONNX_INFERENCE_PATH}/external/protobuf/cmake/libprotobuf-lite.a\n"
                "save\n"
        )

        file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/merge.mri CONTENT "${merge_mri_content}")

        add_custom_command(
                OUTPUT ${ONNX_INFERENCE_PATH}/lib/libonnxruntime.a
                COMMAND ar -M < ${CMAKE_BINARY_DIR}/merge.mri
                DEPENDS ${CMAKE_BINARY_DIR}/merge.mri
                WORKING_DIRECTORY ${ONNX_INFERENCE_PATH}/lib
                COMMENT "Merging static libraries into libonnxruntime.a"
        )

        add_custom_target(merge_libraries ALL DEPENDS ${ONNX_INFERENCE_PATH}/lib/libonnxruntime.a)
    else ()
        message(FATAL_ERROR "[onnx.runtime.sd][E] Unfounded onnxruntime submodule. please clone from https://github.com/microsoft/onnxruntime.git first!")
    endif ()
endmacro()
