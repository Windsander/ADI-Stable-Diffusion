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
macro(auto_link_reference_library target_lib ref_lib_path)
    message(\ ${PROJECT_NAME}=>\ ${Blue}auto_link_reference_library${ColourReset}\ start)

    # compatible caused by NDK find error
    if (ANDROID)
        set(ONNXRUNTIME_LIB ${ref_lib_path}/libonnxruntime.so)
    else()
        find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ref_lib_path})
    endif ()

    if (NOT ONNXRUNTIME_LIB)
        message(FATAL_ERROR "onnxruntime library not found in ${ref_lib_path}")
    else ()
        message(\ ${PROJECT_NAME}=>\ "Found onnxruntime library: ${ONNXRUNTIME_LIB}")
    endif ()

    target_include_directories(
            ${target_lib} PUBLIC
            engine/include
    )
    target_link_libraries(
            ${target_lib} PRIVATE
            ${ONNXRUNTIME_LIB}
    )
    message(\ ${PROJECT_NAME}=>\ ${Blue}auto_link_reference_library${ColourReset}\ done)
endmacro()

#检测生成动态库文件是否存在
macro(check_library_exists lib_name lib_path result_var)
    message(STATUS "Checking library: ${lib_name} in path: ${lib_path}")

    if (WIN32)
        set(LIBRARY_PATH ${lib_path}/${lib_name}.dll)
    elseif (WIN64)
        set(LIBRARY_PATH ${lib_path}/${lib_name}.dll)
    elseif (APPLE)
        set(LIBRARY_PATH ${lib_path}/lib${lib_name}.dylib)
    elseif (LINUX)
        set(LIBRARY_PATH ${lib_path}/lib${lib_name}.so)
    elseif (ANDROID)
        set(LIBRARY_PATH ${lib_path}/lib${lib_name}.so)
    else ()
        set(LIBRARY_PATH ${lib_path}/lib${lib_name}.so)
    endif ()

    message(STATUS "Constructed library path: ${LIBRARY_PATH}")

    if (EXISTS ${LIBRARY_PATH})
        set(${result_var} TRUE)
        message(STATUS "Found ${lib_name} library: ${LIBRARY_PATH}")
    else ()
        set(${result_var} FALSE)
        message(STATUS "${lib_name} library not found at ${LIBRARY_PATH}")
    endif ()
endmacro()

# 资源下载=================================================================================================
# 下载并解压 Define the download_and_decompress function
function(download_and_decompress url filename output_dir)
    file(MAKE_DIRECTORY ${output_dir})
    file(DOWNLOAD ${url} ${output_dir}/${filename} SHOW_PROGRESS)
    if (filename MATCHES ".tgz$" OR filename MATCHES ".tar.gz$")
        execute_process(
                COMMAND tar xzf ${output_dir}/${filename} --strip-components=1
                WORKING_DIRECTORY ${output_dir}
        )
    elseif (filename MATCHES ".aar$")
        execute_process(
                COMMAND unzip -o ${output_dir}/${filename} -d ${output_dir}
        )
    else()
        message(FATAL_ERROR "Unsupported archive format: ${filename}")
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
            list(APPEND ${source_list} ${root_dir}/${sub})
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
    target_include_directories(${library_name} ${include_type} ${root_dir})
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
