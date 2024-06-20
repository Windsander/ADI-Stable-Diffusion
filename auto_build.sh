#!/bin/bash

# Default configuration
DEFAULT_BUILD_TYPE=Debug
DEFAULT_JOBS=8
# shellcheck disable=SC2034
DEFAULT_ANDROID_VER=21          # env used
# shellcheck disable=SC2034
DEFAULT_ANDROID_ABI=arm64-v8a   # env used

# Function: Show help message
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --platform PLATFORM      Target platform (e.g., android, linux, macos, windows)"
    echo "  --build-type TYPE        Build type (Debug, Release, etc.)"
    echo "  --cmake PATH             Path to CMake executable"
    echo "  --ninja PATH             Path to Ninja executable"
    echo "  --jobs N                 Number of parallel jobs"
    echo "  --android-sdk PATH       [android] Path to Android SDK"
    echo "  --android-ndk PATH       [android] Path to Android NDK"
    echo "  --android-ver N          [android] Android system version (default: 21)"
    echo "  --android-abi N          [android] Android ABI (Application Binary Interface, default: arm64-v8a)"
    echo "  -h, --help               Show this help message"
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --platform) PLATFORM="$2"; shift ;;
        --build-type) BUILD_TYPE="$2"; shift ;;
        --cmake) CMAKE="$2"; shift ;;
        --ninja) NINJA="$2"; shift ;;
        --jobs) JOBS="$2"; shift ;;
        --android-sdk) ANDROID_SDK="$2"; shift ;;
        --android-ndk) ANDROID_NDK="$2"; shift ;;
        --android-ver) ANDROID_VER="$2"; shift ;;
        --android-abi) ANDROID_ABI="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Set default values
BUILD_TYPE=${BUILD_TYPE:-$DEFAULT_BUILD_TYPE}
JOBS=${JOBS:-$DEFAULT_JOBS}
CMAKE=${CMAKE:-cmake}
NINJA=${NINJA:-ninja}
ANDROID_VER=${ANDROID_VER:-DEFAULT_ANDROID_VER}
ANDROID_ABI=${ANDROID_ABI:-$DEFAULT_ANDROID_ABI}

# Detect platform if not specified
if [ -z "$PLATFORM" ]; then
    case "$(uname -s)" in
        Linux*)     PLATFORM=linux ;;
        Darwin*)    PLATFORM=macos ;;
        CYGWIN*|MINGW*|MSYS*) PLATFORM=windows ;;
        Android*)   PLATFORM=android ;;
        *)          PLATFORM=unknown ;;
    esac
fi

# Set project root and build directories
PROJECT_ROOT=$(dirname "$0")
BUILD_DIR=${PROJECT_ROOT}/cmake-build-${BUILD_TYPE}-${PLATFORM}

# Clean old build directory
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}

# Platform-specific configuration
case "$PLATFORM" in
    android)
        if [ -z "$ANDROID_SDK" ]; then
            echo "Please set ANDROID_SDK environment variable or pass it as a parameter."
            exit 1
        fi

        if [ -z "$ANDROID_NDK" ]; then
            echo "Please set ANDROID_NDK environment variable or pass it as a parameter."
            exit 1
        fi

        export ANDROID_SDK
        export ANDROID_NDK
        export ANDROID_VER
        export ANDROID_ABI

        TOOLCHAIN_FILE=./apex-toolchain/android-toolchain.cmake #${ANDROID_NDK}/build/cmake/android.toolchain.cmake
        CMAKE_OPTIONS="-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DANDROID_NDK=${ANDROID_NDK}"
        ;;

    linux)
        CMAKE_OPTIONS=""
        ;;

    macos)
        CMAKE_OPTIONS=""
        ;;

    windows)
        CMAKE_OPTIONS=""
        ;;

    *)
        echo "Unknown platform: $PLATFORM"
        exit 1
        ;;
esac

# Run CMake configuration
${CMAKE} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    ${CMAKE_OPTIONS} \
    -S ${PROJECT_ROOT} \
    -B ${BUILD_DIR}

# Check if CMake configuration succeeded
if [ $? -ne 0 ]; then
    echo "CMake configuration failed"
    exit 1
fi

# Run build
${CMAKE} --build ${BUILD_DIR} -- -j${JOBS}

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

echo "Build succeeded"