# aarch64-toolchain.cmake
if(NOT DEFINED ENV{ANDROID_SDK})
    message(FATAL_ERROR "Please set ANDROID_SDK to the path of the Android SDK.")
endif()
if(NOT DEFINED ENV{ANDROID_NDK})
    message(FATAL_ERROR "Please set ANDROID_NDK to the path of the Android NDK.")
endif()
if(NOT DEFINED ENV{ANDROID_VER})
    message(FATAL_ERROR "Please set ANDROID_VER to the Android platform version.")
endif()

# Prepare Android NDK
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION $ENV{ANDROID_VER})
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_ANDROID_ARCH_ABI $ENV{ANDROID_ABI})
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION clang)

set(CMAKE_TOOLCHAIN_FILE $ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake)
set(CMAKE_C_COMPILER $ENV{ANDROID_NDK}/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang)
set(CMAKE_CXX_COMPILER $ENV{ANDROID_NDK}/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++)