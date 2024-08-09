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
elseif (APPLE)
    # macOS use @loader_path
    set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/../lib")
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    message(STATUS "Configuring for macOS")
elseif (LINUX)
    # Linux and Unix-like use $ORIGIN
    set(CMAKE_BUILD_RPATH "$ORIGIN;$ORIGIN/../lib")
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
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
    set(ADI_LIB_PATH "${GENERATED_LIB_PATH}/lib${ADI_LIB}.$<$<BOOL:${ORT_BUILD_SHARED_ADI}>:dylib>$<$<NOT:$<BOOL:${ORT_BUILD_SHARED_ADI}>>:a>")
    set(ORT_LIB_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.$<$<BOOL:${ORT_BUILD_SHARED_ORT}>:dylib>$<$<NOT:$<BOOL:${ORT_BUILD_SHARED_ORT}>>:a>")
elseif (LINUX)
    set(ADI_LIB_PATH "${GENERATED_LIB_PATH}/lib${ADI_LIB}.$<$<BOOL:${ORT_BUILD_SHARED_ADI}>:so>$<$<NOT:$<BOOL:${ORT_BUILD_SHARED_ADI}>>:a>")
    set(ORT_LIB_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.$<$<BOOL:${ORT_BUILD_SHARED_ORT}>:so>$<$<NOT:$<BOOL:${ORT_BUILD_SHARED_ORT}>>:a>")
elseif (ANDROID)
    set(ADI_LIB_PATH "${GENERATED_LIB_PATH}/lib${ADI_LIB}.$<$<BOOL:${ORT_BUILD_SHARED_ADI}>:so>$<$<NOT:$<BOOL:${ORT_BUILD_SHARED_ADI}>>:a>")
    set(ORT_LIB_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.$<$<BOOL:${ORT_BUILD_SHARED_ORT}>:so>$<$<NOT:$<BOOL:${ORT_BUILD_SHARED_ORT}>>:a>")
else()
    message(FATAL_ERROR "[onnx.runtime.sd][E] Unsupported platform!")
endif()

message(STATUS "FOUND_ADI_LIB: ${ADI_LIB_PATH}")
message(STATUS "FOUND_ORT_LIB: ${ORT_LIB_PATH}")

add_executable(${TARGET} main.cc)
target_link_libraries(${TARGET} PRIVATE ${ADI_LIB_PATH} ${ORT_LIB_PATH} ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PUBLIC ${CMAKE_PROJECT_DIR}/include)

if (WIN32)
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying ORT[${ONNX_INFERENCE_VERSION}] library path for Windows"
            COMMAND ${CMAKE_COMMAND} -E copy "${GENERATED_LIB_PATH}/${ORT_LIB}.dll" "$<TARGET_FILE_DIR:${TARGET}>"
    )
elseif (APPLE)
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying ORT[${ONNX_INFERENCE_VERSION}] library path for macOS"
            COMMAND install_name_tool -change "@rpath/lib${ORT_LIB}.${ONNX_INFERENCE_VERSION}.dylib" "@rpath/lib${ORT_LIB}.dylib" "$<TARGET_FILE_DIR:${TARGET}>/${TARGET}"
    )
elseif (LINUX)
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying ORT[${ONNX_INFERENCE_VERSION}] library path for Linux"
            COMMAND patchelf --replace-needed lib${ORT_LIB}.so.${ONNX_INFERENCE_VERSION} ${ORT_LIB}.so "$<TARGET_FILE_DIR:${TARGET}>/${TARGET}"
    )
elseif (ANDROID)
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying ORT[${ONNX_INFERENCE_VERSION}] library path for Android, nothing need to change"
    )
endif()

install(TARGETS ${TARGET} RUNTIME)
install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_CLITOOL_DIR})

# auto_copy_reference_dynamic(${TARGET} onnxruntime ${GENERATED_LIB_PATH} ${CMAKE_CLITOOL_DIR})