# toolchain-linux.cmake

# Set the target system name to Linux
set(CMAKE_SYSTEM_NAME Linux)

# Refer to CMAKE_SYSTEM_PROCESSOR to dynamically set appropriate values
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(TARGET_TRIPLE "x86_64-linux-gnu")
    set(TARGET_ARCH "x86-64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(TARGET_TRIPLE "aarch64-linux-gnu")
    set(TARGET_ARCH "armv8-a")
else()
    message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Function to find and set the compiler
function(find_and_set_compiler)
    foreach(COMPILER ${ARGN})
        message(STATUS "Trying to find compiler: ${COMPILER}")
        find_program(CC_COMPILER NAMES ${COMPILER} PATHS ${CMAKE_PREFIX_PATH})
        if(CC_COMPILER)
            if(${COMPILER} MATCHES "clang")
                set(CMAKE_C_COMPILER "${TARGET_TRIPLE}-clang")
                set(CMAKE_CXX_COMPILER "${TARGET_TRIPLE}-clang++")
                set(CMAKE_LINKER "${TARGET_TRIPLE}-ld")
                set(CMAKE_C_FLAGS "-target ${TARGET_TRIPLE} --sysroot=${SYSROOT_PATH}" CACHE STRING "" FORCE)
                set(CMAKE_CXX_FLAGS "-target ${TARGET_TRIPLE} --sysroot=${SYSROOT_PATH}" CACHE STRING "" FORCE)
            elseif(${COMPILER} MATCHES "gcc")
                set(CMAKE_C_COMPILER "gcc")
                set(CMAKE_CXX_COMPILER "g++")
                set(CMAKE_LINKER "g++")
                set(CMAKE_C_FLAGS "-march=${TARGET_ARCH}" CACHE STRING "" FORCE)
                set(CMAKE_CXX_FLAGS "-march=${TARGET_ARCH}" CACHE STRING "" FORCE)
            elseif(${COMPILER} MATCHES "cl")
                set(CMAKE_C_COMPILER "cl")
                set(CMAKE_CXX_COMPILER "cl")
                set(CMAKE_LINKER "cl")
                # For MSVC, you might need to adjust these flags depending on your setup
                message(FATAL_ERROR "MSVC is not supported for Linux target")
            endif()
            set(CMAKE_COMPILER_FOUND TRUE)
            message(STATUS "Found compiler: ${COMPILER}")
            return()
        endif()
    endforeach()
    message(STATUS "No suitable compiler found from the list: ${CMAKE_COMPILER_FOUND}")
endfunction()

# Set sysroot path for Linux SDK if needed
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(SYSROOT_PATH "${CROSS_COMPILER_PATH}/../sysroot")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    # Assuming sysroot is at a standard location, modify if necessary
    set(SYSROOT_PATH "/usr/local/${TARGET_TRIPLE}/sysroot")
endif()

# Set the correct compiler paths and flags based on the host operating system
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    find_and_set_compiler(clang gcc)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    find_and_set_compiler(clang gcc)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    find_and_set_compiler(clang cl gcc)
else()
    message(FATAL_ERROR "Unsupported CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
endif()
