# toolchain-windows.cmake

# Set the target system name to Windows
set(CMAKE_SYSTEM_NAME Windows)

# Refer to CMAKE_SYSTEM_PROCESSOR to dynamically set appropriate values
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(TARGET_TRIPLE "x86_64-w64-mingw32")
    set(TARGET_ARCH "x86-64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86")
    set(TARGET_TRIPLE "i686-w64-mingw32")
    set(TARGET_ARCH "i686")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(TARGET_TRIPLE "aarch64-w64-mingw32")
    set(TARGET_ARCH "arm64")
else()
    message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Set sysroot path for Windows SDK
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(SYSROOT_PATH "${CROSS_COMPILER_PATH}/../SDKs/Windows")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(SYSROOT_PATH "C:/path/to/Windows/SDK")
endif()

# Set the correct compiler paths and flags based on the host operating system
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    find_cross_compiler(cl clang gcc)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    find_cross_compiler(cl clang gcc)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    find_native_compiler(cl gcc clang)
else()
    message(FATAL_ERROR "Unsupported CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
endif()
