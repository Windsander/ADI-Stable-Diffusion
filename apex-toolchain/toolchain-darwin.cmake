# toolchain-darwin.cmake

# Set the target system name to Darwin
set(CMAKE_SYSTEM_NAME Darwin)

# Refer to CMAKE_SYSTEM_PROCESSOR to dynamically set appropriate values
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(TARGET_TRIPLE "x86_64-apple-darwin")
    set(TARGET_ARCH "x86-64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(TARGET_TRIPLE "arm64-apple-darwin")
    set(TARGET_ARCH "armv8-a")
else()
    message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Set sysroot path for macOS SDK
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(SYSROOT_PATH ${CROSS_COMPILER_PATH}/../SDKs/MacOSX.sdk)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(SYSROOT_PATH /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)
endif()

# Set the correct compiler paths and flags based on the host operating system
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    find_native_compiler(clang gcc)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    find_cross_compiler(clang gcc)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    find_cross_compiler(clang cl gcc)
else()
    message(FATAL_ERROR "Unsupported CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
endif()
