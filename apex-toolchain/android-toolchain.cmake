# android-toolchain.cmake !!!DEPRECATED!!!

# ANDROID_SDK
if(DEFINED ANDROID_SDK)
    set(ANDROID_SDK ${ANDROID_SDK})
elseif(DEFINED ENV{ANDROID_SDK})
    set(ANDROID_SDK $ENV{ANDROID_SDK})
endif()

# ANDROID_NDK
if(DEFINED ANDROID_NDK)
    set(ANDROID_NDK ${ANDROID_NDK})
elseif(DEFINED ENV{ANDROID_NDK})
    set(ANDROID_NDK $ENV{ANDROID_NDK})
endif()

# ANDROID_VER
if(DEFINED ANDROID_VER)
    set(ANDROID_VER ${ANDROID_VER})
elseif(DEFINED ENV{ANDROID_VER})
    set(ANDROID_VER $ENV{ANDROID_VER})
endif()

# Prepare Android NDK
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION clang)
if(DEFINED ANDROID_VER)
    set(CMAKE_SYSTEM_VERSION ${ANDROID_VER})
endif()

# Refer to ANDROID_ABI to dynamically set CMAKE_SYSTEM_PROCESSOR & CMAKE_ANDROID_ARCH_ABI
if(DEFINED ANDROID_ABI)
    if(${ANDROID_ABI} STREQUAL "arm64-v8a")
        set(CMAKE_SYSTEM_PROCESSOR aarch64)
        set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
    elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
        set(CMAKE_SYSTEM_PROCESSOR armv7-a)
        set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
    elseif(${ANDROID_ABI} STREQUAL "x86")
        set(CMAKE_SYSTEM_PROCESSOR x86)
        set(CMAKE_ANDROID_ARCH_ABI x86)
    elseif(${ANDROID_ABI} STREQUAL "x86_64")
        set(CMAKE_SYSTEM_PROCESSOR x86_64)
        set(CMAKE_ANDROID_ARCH_ABI x86_64)
    endif()
endif()

# Set the correct compiler paths based on the host operating system
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(ANDROID_TOOLCHAIN_PATH ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-${CMAKE_SYSTEM_PROCESSOR}/bin)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(ANDROID_TOOLCHAIN_PATH ${ANDROID_NDK}/toolchains/llvm/prebuilt/windows-${CMAKE_SYSTEM_PROCESSOR}/bin)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(ANDROID_TOOLCHAIN_PATH ${ANDROID_NDK}/toolchains/llvm/prebuilt/darwin-${CMAKE_SYSTEM_PROCESSOR}/bin)
endif()

set(CMAKE_C_COMPILER ${ANDROID_TOOLCHAIN_PATH}/clang)
set(CMAKE_CXX_COMPILER ${ANDROID_TOOLCHAIN_PATH}/clang++)

# Include the default Android toolchain file provided by the NDK
set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK}/build/cmake/android.toolchain.cmake)