# Examples CMakeLists.txt

# Basic usage example
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage tiny_llama_cpp)

# Advanced usage example
add_executable(advanced_usage advanced_usage.cpp)
target_link_libraries(advanced_usage tiny_llama_cpp)

# Tokenizer demo example
add_executable(tokenizer_demo tokenizer_demo.cpp)
target_link_libraries(tokenizer_demo tiny_llama_cpp)

# Set output directory for examples
set_target_properties(basic_usage advanced_usage tokenizer_demo
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)