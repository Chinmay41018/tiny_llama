cmake_minimum_required(VERSION 2.8.12)
project(TinyLlamaCpp)

# Set C++14 standard (for older CMake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# GCC specific compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Warning flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    
    # Debug configuration
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    
    # Release configuration with optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    
    # Check GCC version (require 7+)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0")
        message(FATAL_ERROR "GCC version must be at least 7.0")
    endif()
endif()

# Include directories
include_directories(include)

# Source files
set(TINY_LLAMA_SOURCES
    src/tiny_llama.cpp
    src/tokenizer.cpp
    src/model.cpp
    src/attention.cpp
    src/ffn.cpp
    src/matrix.cpp
)

# Create the main library
add_library(tiny_llama_cpp STATIC ${TINY_LLAMA_SOURCES})

# Examples
add_subdirectory(examples)

# Tests (optional, can be disabled)
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()