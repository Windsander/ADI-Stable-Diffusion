#!/bin/bash

# Default build type
DEFAULT_BUILD_TYPE="Debug"

# Check if BUILD_TYPE parameter is provided
if [ -z "$1" ]; then
    BUILD_TYPE=${DEFAULT_BUILD_TYPE}
else
    BUILD_TYPE=$1
    shift  # Remove the first parameter to process other CMake options
fi

# Create the build directory if it doesn't exist
mkdir -p cmake-build-debug

# Run CMake to configure the project and pass all command-line arguments
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} "$@" -S ../clitools -B cmake-build-debug

# Navigate to the build directory
cd cmake-build-debug

# Build the project
cmake --build .