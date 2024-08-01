# toolchain-android.cmake

# Set the target system name to Android
set(CMAKE_SYSTEM_NAME Android)

# Refer to CMAKE_SYSTEM_PROCESSOR to dynamically set CMAKE_SYSTEM_PROCESSOR & CMAKE_ANDROID_ARCH_ABI
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64-v8a")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
    set(TARGET_TRIPLE "aarch64-none-linux-android")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "armeabi-v7a")
    set(CMAKE_SYSTEM_PROCESSOR armv7-a)
    set(CMAKE_ANDROID_ARCH_ABI armeabi-v7a)
    set(TARGET_TRIPLE "armv7-none-linux-androideabi")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86")
    set(CMAKE_SYSTEM_PROCESSOR i686)
    set(CMAKE_ANDROID_ARCH_ABI x86)
    set(TARGET_TRIPLE "i686-none-linux-android")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(CMAKE_SYSTEM_PROCESSOR x86_64)
    set(CMAKE_ANDROID_ARCH_ABI x86_64)
    set(TARGET_TRIPLE "x86_64-none-linux-android")
else()
    message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Set the correct compiler paths based on the host operating system and processor
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(ANDROID_TOOLCHAIN_PATH ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-${CMAKE_HOST_SYSTEM_PROCESSOR}/bin)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(ANDROID_TOOLCHAIN_PATH ${ANDROID_NDK}/toolchains/llvm/prebuilt/windows-${CMAKE_HOST_SYSTEM_PROCESSOR}/bin)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    set(ANDROID_TOOLCHAIN_PATH ${ANDROID_NDK}/toolchains/llvm/prebuilt/darwin-${CMAKE_HOST_SYSTEM_PROCESSOR}/bin)
else()
    message(FATAL_ERROR "Unsupported CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
endif()

set(CMAKE_C_COMPILER ${ANDROID_TOOLCHAIN_PATH}/clang)
set(CMAKE_CXX_COMPILER ${ANDROID_TOOLCHAIN_PATH}/clang++)

# Set appropriate flags for the target architecture
set(CMAKE_C_FLAGS "-target ${TARGET_TRIPLE}${CMAKE_SYSTEM_VERSION} --sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/${CMAKE_HOST_SYSTEM_NAME}-x86_64/sysroot" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "-target ${TARGET_TRIPLE}${CMAKE_SYSTEM_VERSION} --sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/${CMAKE_HOST_SYSTEM_NAME}-x86_64/sysroot" CACHE STRING "" FORCE)
