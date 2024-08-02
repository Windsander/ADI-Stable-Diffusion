# toolchain-android.cmake

# Set the target system name to Android
set(CMAKE_SYSTEM_NAME Android)

# Refer to CMAKE_SYSTEM_PROCESSOR to dynamically set CMAKE_SYSTEM_PROCESSOR & CMAKE_ANDROID_ARCH_ABI
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64-v8a")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
    set(TARGET_TRIPLE "aarch64-linux-android")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "armeabi-v7a")
    set(CMAKE_SYSTEM_PROCESSOR armv7-a)
    set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
    set(TARGET_TRIPLE "armv7a-linux-androideabi")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86")
    set(CMAKE_SYSTEM_PROCESSOR i686)
    set(CMAKE_ANDROID_ARCH_ABI x86)
    set(TARGET_TRIPLE "i686-linux-android")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(CMAKE_SYSTEM_PROCESSOR x86_64)
    set(CMAKE_ANDROID_ARCH_ABI x86_64)
    set(TARGET_TRIPLE "x86_64-linux-android")
else()
    message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Set the correct compiler paths based on the host operating system and processor
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(ANDROID_TOOLCHAIN_PATH ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(ANDROID_TOOLCHAIN_PATH ${ANDROID_NDK}/toolchains/llvm/prebuilt/windows-x86_64)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(ANDROID_TOOLCHAIN_PATH ${ANDROID_NDK}/toolchains/llvm/prebuilt/darwin-x86_64)
else()
    message(FATAL_ERROR "Unsupported CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
endif()

message(STATUS "Found compiler: clang at ${ANDROID_TOOLCHAIN_PATH}/bin/${TARGET_TRIPLE}${CMAKE_SYSTEM_VERSION}-clang[++]")

set(NDK_PROC_aarch64_ABI "arm64-v8a")
set(NDK_PROC_armv7-a_ABI "armeabi-v7a")
set(NDK_PROC_armv6_ABI   "armeabi-v6")
set(NDK_PROC_armv5te_ABI "armeabi")
set(NDK_PROC_i686_ABI    "x86")
set(NDK_PROC_mips_ABI    "mips")
set(NDK_PROC_mips64_ABI  "mips64")
set(NDK_PROC_x86_64_ABI  "x86_64")

set(NDK_ARCH_arm64_ABI  "arm64-v8a")
set(NDK_ARCH_arm_ABI    "armeabi")
set(NDK_ARCH_mips_ABI   "mips")
set(NDK_ARCH_mips64_ABI "mips64")
set(NDK_ARCH_x86_ABI    "x86")
set(NDK_ARCH_x86_64_ABI "x86_64")

# Set appropriate flags for the target architecture
set(CMAKE_C_COMPILER ${ANDROID_TOOLCHAIN_PATH}/bin/clang)
set(CMAKE_CXX_COMPILER ${ANDROID_TOOLCHAIN_PATH}/bin/clang++)
set(CMAKE_C_FLAGS "-target ${TARGET_TRIPLE}${CMAKE_SYSTEM_VERSION} --sysroot=${ANDROID_TOOLCHAIN_PATH}/sysroot" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "-target ${TARGET_TRIPLE}${CMAKE_SYSTEM_VERSION} --sysroot=${ANDROID_TOOLCHAIN_PATH}/sysroot" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")