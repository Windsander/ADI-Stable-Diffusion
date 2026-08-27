cmake_minimum_required(VERSION 3.12)

set(TARGET "adi")
set(GENERATED_LIB_PATH ${MAJOR_SYSTEM_PATH}/lib)
set(CMAKE_CLITOOL_DIR ${MAJOR_SYSTEM_PATH}/bin)
set(CMAKE_PROJECT_DIR ${MAJOR_SYSTEM_PATH}/..)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_SYSTEM_NAME ${MAJOR_SYSTEM_NAME})
set(CMAKE_SYSTEM_PROCESSOR ${MAJOR_SYSTEM_PROCESSOR})
include(${CMAKE_PROJECT_DIR}/apex-toolchain/generic-toolchain.cmake)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CLITOOL_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CLITOOL_DIR})

if (WIN32)
    # Windows not support RPATH
    message(STATUS "Configuring for Windows")
    if (MSVC)
        # windows.h min/max macros break std::min/std::max (C2589); /EHsc for try/catch
        add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
        add_compile_options(/EHsc)
    endif ()
elseif (APPLE)
    # macOS use @loader_path
    set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/../lib;../lib")
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../lib;../lib")
    message(STATUS "Configuring for macOS")
elseif (LINUX)
    # Linux and Unix-like use $ORIGIN
    set(CMAKE_BUILD_RPATH "$ORIGIN;$ORIGIN/../lib;../lib")
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;../lib")
    message(STATUS "Configuring for Unix-like system")
elseif (ANDROID)
    message(STATUS "Android RPATH unavailable")
endif()

# Set the output directory for executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/bin)

# Set the output directory for library files
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/lib)

# Set the output directory for archive files
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/lib)

message(STATUS ${MAJOR_SYSTEM_PROCESSOR} => ${CMAKE_SYSTEM_PROCESSOR})
message(STATUS ${MAJOR_SYSTEM_PATH})

if (ANDROID_NDK)
    message(STATUS "ANDROID_NDK: ${ANDROID_NDK}")
endif ()

PROJECT("adi-cli")

# do prepare project header
include(${CMAKE_PROJECT_DIR}/apex/project_cmake_colors.cmake)
include(${CMAKE_PROJECT_DIR}/apex/project_cmake_static.cmake)
include(${CMAKE_PROJECT_DIR}/apex/project_cmake_ortenv.cmake)
include(${CMAKE_PROJECT_DIR}/apex/project_cmake_utils.cmake)

# compatible caused by NDK find error
set(ADI_LIB adi)
set(ORT_LIB onnxruntime)

# Set library extensions and paths
if (WIN32)
    set(ADI_LIB_PATH "${GENERATED_LIB_PATH}/${ADI_LIB}.lib")
    set(ORT_LIB_PATH "${GENERATED_LIB_PATH}/${ORT_LIB}.lib")
elseif (APPLE)
    if (ORT_BUILD_SHARED_ADI)
        set(ADI_LIB_PATH "${GENERATED_LIB_PATH}/lib${ADI_LIB}.dylib")
    else()
        set(ADI_LIB_PATH "${GENERATED_LIB_PATH}/lib${ADI_LIB}.a")
    endif()

    if (ORT_BUILD_SHARED_ORT)
        set(ORT_LIB_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.dylib")
    else()
        set(ORT_LIB_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.a")
    endif()
elseif (LINUX)
    if (ORT_BUILD_SHARED_ADI)
        set(ADI_LIB_PATH "${GENERATED_LIB_PATH}/lib${ADI_LIB}.so")
    else()
        set(ADI_LIB_PATH "${GENERATED_LIB_PATH}/lib${ADI_LIB}.a")
    endif()

    if (ORT_BUILD_SHARED_ORT)
        set(ORT_LIB_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.so")
    else()
        set(ORT_LIB_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.a")
    endif()
elseif (ANDROID)
    if (ORT_BUILD_SHARED_ADI)
        set(ADI_LIB_PATH "${GENERATED_LIB_PATH}/lib${ADI_LIB}.so")
    else()
        set(ADI_LIB_PATH "${GENERATED_LIB_PATH}/lib${ADI_LIB}.a")
    endif()

    if (ORT_BUILD_SHARED_ORT)
        set(ORT_LIB_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.so")
    else()
        set(ORT_LIB_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.a")
    endif()
else()
    message(FATAL_ERROR "[onnx.runtime.sd][E] Unsupported platform!")
endif()

message(STATUS "FOUND_ADI_LIB: ${ADI_LIB_PATH}")
message(STATUS "FOUND_ORT_LIB: ${ORT_LIB_PATH}")

add_executable(${TARGET} main.cc)
# NOTE: no CMAKE_THREAD_LIBS_INIT on Android — pthread lives in bionic, and a
# misdetected -lpthreads (seen with windows-host NDK toolchains) breaks the
# final link. Android only needs NDK liblog instead (static libadi.a carries
# sentencepiece/protobuf objects that call __android_log_write).
if (ANDROID)
    target_link_libraries(${TARGET} PRIVATE ${ADI_LIB_PATH} ${ORT_LIB_PATH} log)
else ()
    target_link_libraries(${TARGET} PRIVATE ${ADI_LIB_PATH} ${ORT_LIB_PATH} ${CMAKE_THREAD_LIBS_INIT})
endif ()
target_include_directories(${TARGET} PUBLIC ${CMAKE_PROJECT_DIR}/include)

# sentencepiece (stage-C sp tokenizer): header + static lib path from parent build
if (SPM_INCLUDE_DIR)
    target_include_directories(${TARGET} PRIVATE ${SPM_INCLUDE_DIR})
endif ()
if (SPM_STATIC_LIB)
    target_link_libraries(${TARGET} PRIVATE ${SPM_STATIC_LIB})
endif ()

if (WIN32)
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying ORT[${ONNX_INFERENCE_VERSION}] library path for Windows"
            COMMAND ${CMAKE_COMMAND} -E copy "${GENERATED_LIB_PATH}/${ORT_LIB}.dll" "$<TARGET_FILE_DIR:${TARGET}>"
    )
elseif (APPLE)
    # ORT >= 1.19 changed the dylib install-name to @rpath/libonnxruntime.1.dylib
    # (SOVERSION-style); <= 1.18 used @rpath/libonnxruntime.<version>.dylib.
    # Normalize both to the unversioned @rpath/libonnxruntime.dylib we ship in lib/.
    # Non-matching -change rules are silently ignored by install_name_tool.
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying ORT[${ONNX_INFERENCE_VERSION}] library path for macOS"
            COMMAND install_name_tool
                    -change "@rpath/lib${ORT_LIB}.1.dylib" "@rpath/lib${ORT_LIB}.dylib"
                    -change "@rpath/lib${ORT_LIB}.${ONNX_INFERENCE_VERSION}.dylib" "@rpath/lib${ORT_LIB}.dylib"
                    "$<TARGET_FILE_DIR:${TARGET}>/${TARGET}"
    )
elseif (LINUX)
    # ORT >= 1.19 soname is libonnxruntime.so.1; <= 1.18 was libonnxruntime.so.<version>.
    # Normalize DT_NEEDED to the unversioned libonnxruntime.so we ship in lib/
    # (keep the lib prefix). Tolerate either source name not being present.
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying ORT[${ONNX_INFERENCE_VERSION}] library path for Linux"
            COMMAND sh -c "patchelf --replace-needed lib${ORT_LIB}.so.1 lib${ORT_LIB}.so '$<TARGET_FILE_DIR:${TARGET}>/${TARGET>' || true; patchelf --replace-needed lib${ORT_LIB}.so.${ONNX_INFERENCE_VERSION} lib${ORT_LIB}.so '$<TARGET_FILE_DIR:${TARGET}>/${TARGET>' || true"
    )
elseif (ANDROID)
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying ORT[${ONNX_INFERENCE_VERSION}] library path for Android, nothing need to change"
    )
endif()

if (ADI_AUTO_INSTALL)
    install(FILES ${ORT_LIB_PATH} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    install(TARGETS ${TARGET} RUNTIME)

    message(STATUS "Install ort-lib to: ${CMAKE_INSTALL_PREFIX}/lib")
    message(STATUS "Install adi-cli to: ${CMAKE_INSTALL_PREFIX}/bin")
endif ()

install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_CLITOOL_DIR})

# auto_copy_reference_dynamic(${TARGET} onnxruntime ${GENERATED_LIB_PATH} ${CMAKE_CLITOOL_DIR})