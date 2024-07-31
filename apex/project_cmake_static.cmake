# Defines functions and macros useful for ORT CMakeLists using.
# Created:
#
# - by Arikan.Li on 2024/06/01.
#
# Note:
#
# - ???What?

# StaticLib==============================================================================================
# Auto Choose Archiver based on platforms to generate onnxruntime.a
function(auto_choose_archiver_for_ort)
    if (ORT_BUILD_SHARED_ORT)
        return()
    endif()
    message(\ ${PROJECT_NAME}=>\ ${Blue}auto_choose_archiver${ColourReset}\ start)

    # Check if ARCHIVER_PATH is already set
    if (NOT ARCHIVER_PATH)
        # compatible caused by NDK find error
        if (WIN32)
            set(ARCHIVER_PATH lib)
        elseif (APPLE)
            set(ARCHIVER_PATH libtool)
        elseif (LINUX OR ANDROID)
            set(ARCHIVER_PATH ar)
        else()
            message(FATAL_ERROR "[onnx.runtime.sd][E] Unsupported platform!")
        endif()
    endif()

    if (NOT ARCHIVER_PATH)
        message(FATAL_ERROR "${ARCHIVER_PATH} library not found!")
    else()
        message(\ ${PROJECT_NAME}=>\ "auto selected ARCHIVER: ${ARCHIVER_PATH}")
    endif()

    message(\ ${PROJECT_NAME}=>\ ${Blue}auto_choose_archiver${ColourReset}\ done)
endfunction()

# Build Windows Static ORT.a
macro(merge_static_libs_windows)
    set(TEMP_SCRIPT "${ONNX_INFERENCE_PATH}/run_lib.bat")
    file(WRITE ${TEMP_SCRIPT} "@echo off\n")
    file(APPEND ${TEMP_SCRIPT} "${ARCHIVER_PATH} /OUT:libonnxruntime.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_common.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_framework.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_flatbuffers.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_graph.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_mlas.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_optimizer.lib \\\n")
    if (ORT_ENABLE_CUDA)
        file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_providers_cuda.lib \\\n")
    endif ()
    if (ORT_ENABLE_TENSORRT)
        file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_providers_tensorrt.lib \\\n")
    endif ()
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/onnx-build/libonnx_proto.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/onnx-build/libonnx.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_providers.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_session.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_util.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/lib/libgmock.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/lib/libgtest.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/re2-build/libre2.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/pytorch_cpuinfo-build/libcpuinfo.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/google_nsync-build/libnsync_cpp.lib \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/protobuf-build/libprotobuf-lited.lib\n")

    execute_process(
            COMMAND ${TEMP_SCRIPT}
            WORKING_DIRECTORY ${ONNX_INFERENCE_PATH}
            RESULT_VARIABLE result
            ERROR_VARIABLE error
    )
    message(STATUS "[onnx.runtime.sd][I] execute_process build libonnxruntime.lib done")

    if (result)
        message(FATAL_ERROR "Failed to execute lib: ${error}")
    endif()
endmacro()

# Build Apple Static ORT.a
macro(merge_static_libs_macos)
    set(TEMP_SCRIPT "${ONNX_INFERENCE_PATH}/run_libtool.sh")
    file(WRITE ${TEMP_SCRIPT} "#!/bin/sh\n")
    file(APPEND ${TEMP_SCRIPT} "${ARCHIVER_PATH} -static -o libonnxruntime.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_common.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_framework.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_flatbuffers.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_graph.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_mlas.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_optimizer.a \\\n")
    if (ORT_ENABLE_COREML)
        file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_providers_coreml.a \\\n")
    endif ()
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_providers.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_session.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/libonnxruntime_util.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/onnx-build/libonnx_proto.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/onnx-build/libonnx.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/lib/libgmock.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/lib/libgtest.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/re2-build/libre2.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/pytorch_cpuinfo-build/libcpuinfo.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/google_nsync-build/libnsync_cpp.a \\\n")
    file(APPEND ${TEMP_SCRIPT} "${ONNX_INFERENCE_PATH}/_deps/protobuf-build/libprotobuf-lited.a\n")

    if (UNIX)
        execute_process(COMMAND chmod +x ${TEMP_SCRIPT})
    endif()
    execute_process(
            COMMAND ${TEMP_SCRIPT}
            WORKING_DIRECTORY ${ONNX_INFERENCE_PATH}
            RESULT_VARIABLE result
            ERROR_VARIABLE error
    )
    message(STATUS "[onnx.runtime.sd][I] execute_process build libonnxruntime.a done")

    if (result)
        message(FATAL_ERROR "Failed to execute libtool: ${error}")
    endif()
endmacro()

# Build Android Static ORT.a
macro(merge_static_libs_android)
    message( "[onnx.runtime.sd][I] auto_merge_submodule_compiled ${ONNX_INFERENCE_PATH}/merge.mri")
    if (NOT DEFINED ARCHIVER_PATH OR ARCHIVER_PATH STREQUAL "")
        message(FATAL_ERROR "[onnx.runtime.sd][E] ARCHIVER_PATH is not set. Please set ARCHIVER_PATH to the path of the archiver tool.")
    endif()
    if(NOT DEFINED ONNX_INFERENCE_PATH OR ONNX_INFERENCE_PATH STREQUAL "")
        message(FATAL_ERROR "[onnx.runtime.sd][E] ONNX_INFERENCE_PATH is not set or is empty")
    endif()

    # touch merge.mri temp
    string(
            CONCAT merge_mri_content
            "create libonnxruntime.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_common.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_framework.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_flatbuffers.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_graph.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_mlas.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_optimizer.a\n"
    )
    if (ORT_ENABLE_NNAPI)
        string(
                APPEND merge_mri_content
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_providers_nnapi.a\n"
        )
    endif ()
    string(
            APPEND merge_mri_content
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_providers.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_session.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_util.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnx_proto.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnx.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/lib/libgmock.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/lib/libgtest.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/_deps/re2-build/libre2.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/_deps/pytorch_cpuinfo-build/libcpuinfo.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/_deps/google_nsync-build/libnsync_cpp.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/_deps/protobuf-build/libprotobuf-lited.a\n"
            "save\n"
    )
    file(WRITE ${ONNX_INFERENCE_PATH}/merge.mri ${merge_mri_content})

    message(STATUS "[onnx.runtime.sd][I] merge.mri input:\n${Cyan}${merge_mri_content}${ColourReset}")

    set(TEMP_SCRIPT "${ONNX_INFERENCE_PATH}/run_ar.sh")
    file(WRITE ${TEMP_SCRIPT} "#!/bin/sh\n")
    file(APPEND ${TEMP_SCRIPT} "${ARCHIVER_PATH} -M < ${ONNX_INFERENCE_PATH}/merge.mri\n")
    if (UNIX)
        execute_process(COMMAND chmod +x ${TEMP_SCRIPT})
    endif()
    execute_process(
            COMMAND ${TEMP_SCRIPT}
            WORKING_DIRECTORY ${ONNX_INFERENCE_PATH}
            RESULT_VARIABLE result
            ERROR_VARIABLE error
    )
    message(STATUS "[onnx.runtime.sd][I] execute_process build ${ARCHIVER_PATH} ONNXRuntime.a done")

    if (result)
        message(FATAL_ERROR "Failed to execute ar: ${error}")
    endif()
endmacro()

# Build Linux Static ORT.a
macro(merge_static_libs_linux)
    message( "[onnx.runtime.sd][I] auto_merge_submodule_compiled ${ONNX_INFERENCE_PATH}/merge.mri")
    if (NOT DEFINED ARCHIVER_PATH OR ARCHIVER_PATH STREQUAL "")
        message(FATAL_ERROR "[onnx.runtime.sd][E] ARCHIVER_PATH is not set. Please set ARCHIVER_PATH to the path of the archiver tool.")
    endif()
    if(NOT DEFINED ONNX_INFERENCE_PATH OR ONNX_INFERENCE_PATH STREQUAL "")
        message(FATAL_ERROR "[onnx.runtime.sd][E] ONNX_INFERENCE_PATH is not set or is empty")
    endif()

    # touch merge.mri temp
    string(
            CONCAT merge_mri_content
            "create libonnxruntime.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_common.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_framework.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_flatbuffers.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_graph.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_mlas.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_optimizer.a\n"
    )
    if (ORT_ENABLE_CUDA)
        string(
                APPEND merge_mri_content
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_providers_cuda.a\n"
        )
    endif ()
    if (ORT_ENABLE_TENSORRT)
        string(
                APPEND merge_mri_content
                "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_providers_tensorrt.a\n"
        )
    endif ()
    string(
            APPEND merge_mri_content
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_providers.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_session.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnxruntime_util.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnx_proto.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/libonnx.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/lib/libgmock.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/lib/libgtest.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/_deps/re2-build/libre2.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/_deps/pytorch_cpuinfo-build/libcpuinfo.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/_deps/google_nsync-build/libnsync_cpp.a\n"
            "addlib ${ONNX_INFERENCE_PATH}/_deps/protobuf-build/libprotobuf-lited.a\n"
            "save\n"
    )
    file(WRITE ${ONNX_INFERENCE_PATH}/merge.mri ${merge_mri_content})

    message(STATUS "[onnx.runtime.sd][I] merge.mri input:\n${Cyan}${merge_mri_content}${ColourReset}")

    set(TEMP_SCRIPT "${ONNX_INFERENCE_PATH}/run_ar.sh")
    file(WRITE ${TEMP_SCRIPT} "#!/bin/sh\n")
    file(APPEND ${TEMP_SCRIPT} "${ARCHIVER_PATH} -M < ${ONNX_INFERENCE_PATH}/merge.mri\n")
    if (UNIX)
        execute_process(COMMAND chmod +x ${TEMP_SCRIPT})
    endif()
    execute_process(
            COMMAND ${TEMP_SCRIPT}
            WORKING_DIRECTORY ${ONNX_INFERENCE_PATH}
            RESULT_VARIABLE result
            ERROR_VARIABLE error
    )
    message(STATUS "[onnx.runtime.sd][I] execute_process build ${ARCHIVER_PATH} ONNXRuntime.a done")

    if (result)
        message(FATAL_ERROR "Failed to execute ar: ${error}")
    endif()
endmacro()
