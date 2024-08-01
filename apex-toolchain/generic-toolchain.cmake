# CustomToolchain.cmake

# Check required variables
if(NOT DEFINED CMAKE_SYSTEM_NAME)
    message(FATAL_ERROR "CMAKE_SYSTEM_NAME is not set")
endif()

if(NOT DEFINED CMAKE_SYSTEM_PROCESSOR)
    message(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR is not set")

endif()
if(NOT DEFINED CMAKE_HOST_SYSTEM_NAME)
    execute_process(COMMAND uname -s OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT DEFINED CMAKE_HOST_SYSTEM_PROCESSOR)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_PROCESSOR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Set compiler and flags based on system and processor
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    include(${CMAKE_CURRENT_SOURCE_DIR}/apex-toolchain/toolchain-android.cmake)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    include(${CMAKE_CURRENT_SOURCE_DIR}/apex-toolchain/toolchain-linux.cmake)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    include(${CMAKE_CURRENT_SOURCE_DIR}/apex-toolchain/toolchain-darwin.cmake)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    include(${CMAKE_CURRENT_SOURCE_DIR}/apex-toolchain/toolchain-windows.cmake)
else()
    message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}")
endif()