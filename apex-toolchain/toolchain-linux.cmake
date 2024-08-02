# toolchain-linux.cmake

# Set the target system name to Linux
set(CMAKE_SYSTEM_NAME Linux)

# Refer to CMAKE_SYSTEM_PROCESSOR to dynamically set appropriate values
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(TARGET_TRIPLE "x86_64-linux-gnu")
    set(TARGET_ARCH "x86-64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(TARGET_TRIPLE "aarch64-linux-gnu")
    set(TARGET_ARCH "aarch64")
else()
    message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Set the correct compiler paths and flags based on the host operating system
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    find_cross_compiler(gcc clang)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    find_native_compiler(gcc clang)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    find_cross_compiler(gcc clang)
else()
    message(FATAL_ERROR "Unsupported CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
endif()
