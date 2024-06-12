#!/bin/bash

# Default values for VERSION and PLATFORM
DEFAULT_VERSION="v1.18.0"
DEFAULT_PLATFORM="osx-arm64"

# Use provided VERSION and PLATFORM or default values if not provided
VERSION=${1:-$DEFAULT_VERSION}
PLATFORM=${2:-$DEFAULT_PLATFORM}

# Get the name of the extracted folder
ORT_FOLDER="onnxruntime-${PLATFORM}-${VERSION}"

# Construct the download URL and filename
URL="https://github.com/microsoft/onnxruntime/releases/download/${VERSION}/${ORT_FOLDER}.tgz"
FILENAME="onnxruntime-${PLATFORM}-${VERSION}.tgz"

# Create the ./engine directory if it doesn't exist
mkdir -p ./engine

# Download the file to the ./engine directory
echo "Downloading ${URL}..."
curl -L -o ./engine/${FILENAME} ${URL}

# Extract the file to the ./engine directory
echo "Extracting ${FILENAME}..."
tar -xzvf ./engine/${FILENAME} -C ./engine

echo "Done."

