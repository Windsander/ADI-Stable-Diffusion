# Defines functions and macros useful for common CMakeLists using.
# Created:
#
# - by Arikan.Li on 2021/12/31.
#
# Note:
#
# - ???What?

# 平台限定=================================================================================================
#自动检测关联项目 onnxruntime 是否以在本地
macro(auto_check_reference_submodule)
    # 动态检测 ./engine/onnxruntime 是否存在
    if (NOT EXISTS ${ONNX_PATH}/onnxruntime)
        message(STATUS "onnxruntime submodule not found. Cloning from GitHub...")
        execute_process(
                COMMAND git clone --recursive https://github.com/microsoft/onnxruntime.git ${ONNX_PATH}/onnxruntime
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE result
        )
        if (result)
            message(FATAL_ERROR "Failed to clone onnxruntime repository")
        endif ()
    else ()
        message(STATUS "Found onnxruntime submodule.")
    endif ()
endmacro()

#自动检测关联项目 onnxruntime 是否以在本地
macro(auto_build_reference_submodule)
    # 动态检测 ./engine/onnxruntime 是否存在
    if (EXISTS ${ONNX_PATH}/onnxruntime)
        # do prepare option auto enable, based on paltform
        if (WIN32)                          # for Windows x32
            execute_process(
                    COMMAND ${ONNX_PATH}/onnxruntime/build.bat --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
                    WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
                    RESULT_VARIABLE ${ONNX_PATH}/result
            )
            if (result)
                message(FATAL_ERROR "Failed to build WIN32 ONNXRuntime")
            endif ()
        elseif (WIN64)                      # for Windows x64
            execute_process(
                    COMMAND ${ONNX_PATH}/onnxruntime/build.bat -–arm64 --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
                    WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
                    RESULT_VARIABLE ${ONNX_PATH}/result
            )
            if (result)
                message(FATAL_ERROR "Failed to build WIN64 ONNXRuntime")
            endif ()
        elseif (APPLE)                      # for MacOS X or iOS, watchOS, tvOS (since 3.10.3)
            execute_process(
                    COMMAND ${ONNX_PATH}/onnxruntime/build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES="x86_64;arm64"
                    WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
                    RESULT_VARIABLE ${ONNX_PATH}/result
            )
            if (result)
                message(FATAL_ERROR "Failed to build iOS/OSX ONNXRuntime")
            endif ()
        elseif (UNIX AND NOT APPLE)         # for Linux, BSD, Solaris, Minix
            execute_process(
                    COMMAND ${ONNX_PATH}/onnxruntime/build.sh --config RelWithDebInfo --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
                    WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
                    RESULT_VARIABLE ${ONNX_PATH}/result
            )
            if (result)
                message(FATAL_ERROR "Failed to build Linux ONNXRuntime")
            endif ()
        elseif (ANDROID)                    # for Android
            # 执行 build.sh 脚本来生成 Android-ONNXRuntime
            execute_process(
                    COMMAND ${ONNX_PATH}/onnxruntime/build.sh --android --android_sdk_path ${ANDROID_SDK} --android_ndk_path ${ANDROID_NDK} --android_abi ${CMAKE_ANDROID_ARCH_ABI} --android_api ${CMAKE_SYSTEM_VERSION}
                    WORKING_DIRECTORY ${ONNX_PATH}/onnxruntime
                    RESULT_VARIABLE ${ONNX_PATH}/result
            )
            if (result)
                message(FATAL_ERROR "Failed to build Android ONNXRuntime")
            endif ()
        endif ()
    else ()
        message(STATUS "Unfounded onnxruntime submodule. please clone from https://github.com/microsoft/onnxruntime.git first!")
    endif ()
endmacro()
