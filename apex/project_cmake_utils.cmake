# Defines functions and macros useful for common CMakeLists using.
# Created:
#
# - by Arikan.Li on 2021/12/31.
#
# Note:
#
# - ???What?

# 平台限定=================================================================================================
#自动连接对应平台关联库
macro(auto_link_reference_library target_lib lib_name ref_lib_path)
    message(\ ${PROJECT_NAME}=>\ ${Blue}auto_link_reference_library${ColourReset}\ start)

    # compatible caused by NDK find error
    if (WIN32)
        if (ORT_BUILD_SHARED_LIBS)
            set(LIBRARY_PATH "${ref_lib_path}/${lib_name}.dll")
        else()
            set(LIBRARY_PATH "${ref_lib_path}/${lib_name}.lib")
        endif()
        set(INCLUDE_PATH ${ONNX_INFERENCE_PATH}/include)
    elseif (APPLE)
        if (ORT_BUILD_SHARED_LIBS)
            set(LIBRARY_PATH "${ref_lib_path}/lib${lib_name}.dylib")
        else()
            set(LIBRARY_PATH "${ref_lib_path}/lib${lib_name}.a")
        endif()
        set(INCLUDE_PATH ${ONNX_INFERENCE_PATH}/include)
    elseif (LINUX)
        if (ORT_BUILD_SHARED_LIBS)
            set(LIBRARY_PATH "${ref_lib_path}/lib${lib_name}.so")
        else()
            set(LIBRARY_PATH "${ref_lib_path}/lib${lib_name}.a")
        endif()
        set(INCLUDE_PATH ${ONNX_INFERENCE_PATH}/include)
    elseif (ANDROID)
        if (ORT_BUILD_SHARED_LIBS)
            set(LIBRARY_PATH "${ref_lib_path}/lib${lib_name}.so")
        else()
            set(LIBRARY_PATH "${ref_lib_path}/lib${lib_name}.a")
        endif()
        set(INCLUDE_PATH ${ONNX_INFERENCE_PATH}/headers)
    else()
        message(FATAL_ERROR "[onnx.runtime.sd][E] Unsupported platform!")
    endif()

    if (NOT LIBRARY_PATH)
        message(FATAL_ERROR "${lib_name} library not found in ${ref_lib_path}")
    else()
        message(\ ${PROJECT_NAME}=>\ "Found ${lib_name} library: ${LIBRARY_PATH}")
        message(\ ${PROJECT_NAME}=>\ "Found ${lib_name} include: ${INCLUDE_PATH}")
    endif()

    target_include_directories(
            ${target_lib} PUBLIC
            ${INCLUDE_PATH}
    )
    target_link_libraries(
            ${target_lib} PRIVATE
            ${LIBRARY_PATH}
    )
    message(\ ${PROJECT_NAME}=>\ ${Blue}auto_link_reference_library${ColourReset}\ done)
endmacro()

#检测生成动态库文件是否存在
macro(check_library_exists lib_name lib_path result_var)
    message(STATUS "Checking library: ${lib_name} in path: ${lib_path}")

    if (WIN32)
        if (ORT_BUILD_SHARED_LIBS)
            set(LIBRARY_PATH "${lib_path}/${lib_name}.dll")
        else()
            set(LIBRARY_PATH "${lib_path}/${lib_name}.lib")
        endif()
    elseif (APPLE)
        if (ORT_BUILD_SHARED_LIBS)
            set(LIBRARY_PATH "${lib_path}/lib${lib_name}.dylib")
        else()
            set(LIBRARY_PATH "${lib_path}/lib${lib_name}.a")
        endif()
    elseif (LINUX)
        if (ORT_BUILD_SHARED_LIBS)
            set(LIBRARY_PATH "${lib_path}/lib${lib_name}.so")
        else()
            set(LIBRARY_PATH "${lib_path}/lib${lib_name}.a")
        endif()
    elseif (ANDROID)
        if (ORT_BUILD_SHARED_LIBS)
            set(LIBRARY_PATH "${lib_path}/lib${lib_name}.so")
        else()
            set(LIBRARY_PATH "${lib_path}/lib${lib_name}.a")
        endif()
    else()
        message(FATAL_ERROR "[onnx.runtime.sd][E] Unsupported platform!")
    endif()

    message(STATUS "Constructed library path: ${LIBRARY_PATH}")

    if (EXISTS ${LIBRARY_PATH})
        set(${result_var} TRUE)
        message(STATUS "Found ${lib_name} library: ${LIBRARY_PATH}")
    else()
        set(${result_var} FALSE)
        message(STATUS "${lib_name} library not found at ${LIBRARY_PATH}")
    endif()
endmacro()

# 资源下载=================================================================================================
# 下载并解压 Define the download_and_decompress function
function(download_and_decompress url filename output_dir)
    file(MAKE_DIRECTORY ${output_dir})

    # Check if the platform is Linux and filename contains x86_64
    if (LINUX AND filename MATCHES "x86_64")
        string(REPLACE "x86_64" "x64" modified_filename ${filename})
        string(REPLACE ${filename} ${modified_filename} download_url ${url})
        message(STATUS "Downloading ${download_url} as ${filename}")
        file(DOWNLOAD ${download_url} ${output_dir}/${filename} SHOW_PROGRESS)
    else()
        file(DOWNLOAD ${url} ${output_dir}/${filename} SHOW_PROGRESS)
    endif()

    if (filename MATCHES ".tgz$" OR filename MATCHES ".tar.gz$")
        execute_process(
                COMMAND tar xzf ${output_dir}/${filename} --strip-components=1
                WORKING_DIRECTORY ${output_dir}
        )
    elseif (filename MATCHES ".aar$")
        execute_process(
                COMMAND unzip -o ${output_dir}/${filename} -d ${output_dir}
        )
    elseif (filename MATCHES ".zip$")
        execute_process(
                COMMAND unzip -o ${output_dir}/${filename} -d ${output_dir}
        )
    else()
        message(FATAL_ERROR "Unsupported archive format: ${filename}")
    endif()

    # Rename the directory if necessary
    if (filename MATCHES "cuda12")
        file(GLOB extracted_dirs LIST_DIRECTORIES true "${output_dir}/*")
        foreach(dir ${extracted_dirs})
            if (IS_DIRECTORY ${dir} AND NOT dir MATCHES "cuda12")
                # Extract the base name and version
                get_filename_component(dir_name ${dir} NAME)
                string(REGEX REPLACE "(.*)-([0-9]+\\.[0-9]+\\.[0-9]+)" "\\1-cuda12-\\2" new_dir_name ${dir_name})
                file(RENAME ${dir} "${output_dir}/${new_dir_name}")
            endif()
        endforeach()
    endif()

    file(REMOVE ${output_dir}/${filename})
endfunction()

# 资源遍历=================================================================================================
# 自动添加库索引关联子文件
function(auto_include_all_files root_dir)
    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
    foreach (sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub}
                AND NOT (${root_dir} MATCHES ".*/prebuilt.*")
                AND NOT (${root_dir} MATCHES ".*/CMakeFiles.*"))
            auto_include_all_files(${root_dir}/${sub})
        elseif (NOT (${root_dir} MATCHES ".*/test.*")
                AND NOT (${sub} MATCHES ".DS_Store"))
            continue()
        endif ()
    endforeach ()
    message(\ ${PROJECT_NAME}=>\ auto_include_all_files::\ ${root_dir})
    include_directories(${root_dir})
endfunction()

# 指定资源列表，自动遍历指定目录所有目录，添加入表中
macro(auto_target_sources source_list path_dir root_dir)
    file(GLOB AUTO_SOURCE_SUB RELATIVE ${path_dir}/${root_dir} ${path_dir}/${root_dir}/*)
    foreach (sub ${AUTO_SOURCE_SUB})
        if (IS_DIRECTORY ${path_dir}/${root_dir}/${sub}
                AND (${sub} MATCHES "adjustment"))
            auto_choose_platform_adjusts(${source_list} ${root_dir}/${sub})
        elseif (IS_DIRECTORY ${path_dir}/${root_dir}/${sub}
                AND (${sub} MATCHES "shader"))
            auto_choose_platform_shaders(${source_list} ${root_dir}/${sub})
        elseif (IS_DIRECTORY ${path_dir}/${root_dir}/${sub}
                AND (${sub} MATCHES "driver"))
            auto_choose_platform_drivers(${source_list} ${root_dir}/${sub})
        elseif (IS_DIRECTORY ${path_dir}/${root_dir}/${sub}
                AND (${sub} MATCHES "environment"))
            auto_choose_platform_envs(${source_list} ${root_dir}/${sub})
        elseif (IS_DIRECTORY ${path_dir}/${root_dir}/${sub}
                AND NOT (${path_dir}/${root_dir} MATCHES ".*/prebuilt.*")
                AND NOT (${path_dir}/${root_dir} MATCHES ".*/CMakeFiles.*"))
            auto_target_sources(${source_list} ${path_dir} ${root_dir}/${sub})
        elseif (NOT (${path_dir}/${root_dir} MATCHES ".*/test.*")
                AND NOT (${sub} MATCHES ".DS_Store"))
            message(\ ${PROJECT_NAME}\ =>\ auto_target_sources::\ ${sub})
            list(APPEND ${source_list} ${path_dir}/${root_dir}/${sub})
        endif ()
    endforeach ()
endmacro()

# 指定库，自动添加库索引关联子文件
macro(auto_target_include library path_dir root_dir include_type)
    file(GLOB AUTO_INCLUDE_SUB RELATIVE ${path_dir}/${root_dir} ${path_dir}/${root_dir}/*)
    foreach (sub ${AUTO_INCLUDE_SUB})
        if (IS_DIRECTORY ${path_dir}/${root_dir}/${sub}
                AND NOT (${path_dir}/${root_dir} MATCHES ".*/prebuilt.*")
                AND NOT (${path_dir}/${root_dir} MATCHES ".*/CMakeFiles.*"))
            auto_target_include(${library} ${path_dir} ${root_dir}/${sub} ${include_type})
        elseif (NOT (${path_dir}/${root_dir} MATCHES ".*/test.*")
                AND NOT (${sub} MATCHES ".DS_Store"))
            continue()
        endif ()
    endforeach ()
    message(\ ${PROJECT_NAME}=>\ auto_target_include::\ ${BoldMagenta}${include_type}${ColourReset} \ ${root_dir})
    target_include_directories(${library_name} ${include_type} ${path_dir}/${root_dir})
endmacro()

# 指定库，自动添加关联子文件
macro(auto_include path_dir root_dir)
    file(GLOB AUTO_INCLUDE_SUB RELATIVE ${path_dir}/${root_dir} ${path_dir}/${root_dir}/*)
    foreach (sub ${AUTO_INCLUDE_SUB})
        if (IS_DIRECTORY ${path_dir}/${root_dir}/${sub}
                AND NOT (${path_dir}/${root_dir} MATCHES ".*/prebuilt.*")
                AND NOT (${path_dir}/${root_dir} MATCHES ".*/CMakeFiles.*"))
            auto_include(${path_dir} ${root_dir}/${sub})
        elseif (NOT (${path_dir}/${root_dir} MATCHES ".*/test.*")
                AND NOT (${sub} MATCHES ".DS_Store"))
            continue()
        endif ()
    endforeach ()
    message(\ ${PROJECT_NAME}=>\ auto_include::\ ${path_dir}/${root_dir})
    include_directories(${path_dir}/${root_dir})
endmacro()

# 通用列表数据打印
macro(auto_print_list source_list)
    message("${BoldMagenta}")
    message(\ ${PROJECT_NAME}\ =>\ All_Sources::\ ${source_list})
    foreach (sub ${${source_list}})
        message("\            =>\ ${sub}")
    endforeach ()
    message(\ ${PROJECT_NAME}\ =>\ All_Sources::\ Done)
    message("${ColourReset}")
endmacro()

# 资源归档=================================================================================================
# 前置路径归档cmake函数定义
function(assign_source_group files)
    foreach (_source ${files})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CORE_FRAMEWORK_PATH}" "${_source}")
        else ()
            set(_source_rel "${_source}")
        endif ()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach ()
endfunction(assign_source_group)

function(directories_recursively_assign_source_group root_dir)
    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
    foreach (sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})
            directories_recursively_assign_source_group(${root_dir}/${sub})
        else ()
            assign_source_group(${root_dir}/${sub})
        endif ()
    endforeach ()
endfunction()
