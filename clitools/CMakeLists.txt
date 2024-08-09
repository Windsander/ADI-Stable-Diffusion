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

PROJECT("adi-cli")

# do prepare project header
include(${CMAKE_PROJECT_DIR}/apex/project_cmake_colors.cmake)
include(${CMAKE_PROJECT_DIR}/apex/project_cmake_static.cmake)
include(${CMAKE_PROJECT_DIR}/apex/project_cmake_ortenv.cmake)
include(${CMAKE_PROJECT_DIR}/apex/project_cmake_utils.cmake)

# compatible caused by NDK find error
find_library(ADI_LIB NAMES adi PATHS ${GENERATED_LIB_PATH})
set(ORT_LIB onnxruntime)
if (WIN32)
    if (ORT_BUILD_SHARED_ORT)
        set(LIBRARY_PATH "${GENERATED_LIB_PATH}/${ORT_LIB}.lib")     # dll:: linking should be .lib as import library(.lib in stub mode)
    else()
        set(LIBRARY_PATH "${GENERATED_LIB_PATH}/${ORT_LIB}.lib")     # lib:: make sure .lib is not a stub format, static needs full code
    endif()
    set(INCLUDE_PATH ${ONNX_INFERENCE_PATH}/include)
elseif (APPLE)
    if (ORT_BUILD_SHARED_ORT)
        set(LIBRARY_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.dylib")
    else()
        set(LIBRARY_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.a")
    endif()
    set(INCLUDE_PATH ${ONNX_INFERENCE_PATH}/include)
elseif (LINUX)
    if (ORT_BUILD_SHARED_ORT)
        set(LIBRARY_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.so")
    else()
        set(LIBRARY_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.a")
    endif()
    set(INCLUDE_PATH ${ONNX_INFERENCE_PATH}/include)
elseif (ANDROID)
    if (ORT_BUILD_SHARED_ORT)
        set(LIBRARY_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.so")
    else()
        set(LIBRARY_PATH "${GENERATED_LIB_PATH}/lib${ORT_LIB}.a")
    endif()
    set(INCLUDE_PATH ${ONNX_INFERENCE_PATH}/headers)
else()
    message(FATAL_ERROR "[onnx.runtime.sd][E] Unsupported platform!")
endif()

add_executable(${TARGET} main.cc)
target_link_libraries(${TARGET} PRIVATE ${ADI_LIB} ${LIBRARY_PATH})
target_include_directories(${TARGET} PUBLIC ${CMAKE_PROJECT_DIR}/include)

if (APPLE)
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying library path for macOS"
            COMMAND install_name_tool -change "@rpath/lib${ORT_LIB}.${ONNX_INFERENCE_VERSION}.dylib" "@rpath/lib${ORT_LIB}.dylib" "$<TARGET_FILE_DIR:${TARGET}>/${TARGET}"
    )
elseif (UNIX)
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying library path for Linux"
            COMMAND patchelf --replace-needed lib${ORT_LIB}.so.${ONNX_INFERENCE_VERSION} ${ORT_LIB}.so "$<TARGET_FILE_DIR:${TARGET}>/${TARGET}"
    )
elseif (WIN32)
    add_custom_command(
            TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Modifying library path for Windows"
            COMMAND ${CMAKE_COMMAND} -E copy "${GENERATED_LIB_PATH}/${ORT_LIB}.dll" "$<TARGET_FILE_DIR:${TARGET}>"
    )
endif()

install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_CLITOOL_DIR})

# auto_copy_reference_dynamic(${TARGET} onnxruntime ${GENERATED_LIB_PATH} ${CMAKE_CLITOOL_DIR})