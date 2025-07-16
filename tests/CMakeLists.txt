# Tests CMakeLists.txt

# Main test executable
add_executable(run_tests
    test_main.cpp
    test_tokenizer.cpp
    test_model.cpp
    test_matrix.cpp
    test_integration.cpp
    test_exceptions.cpp
    test_attention.cpp
    test_ffn.cpp
    test_transformer_block.cpp
    test_tiny_llama_api.cpp
    test_api_error_handling.cpp
)

target_link_libraries(run_tests tiny_llama_cpp)

# Set output directory for tests
set_target_properties(run_tests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Individual test drivers
add_executable(vocabulary_test_driver
    vocabulary_test_driver.cpp
    test_tokenizer.cpp
)

add_executable(tokenizer_test_driver
    tokenizer_test_driver.cpp
    test_tokenizer.cpp
)

add_executable(ffn_test_driver
    ffn_test_driver.cpp
    test_ffn.cpp
)

add_executable(transformer_block_test_driver
    transformer_block_test_driver.cpp
    test_transformer_block.cpp
)

add_executable(model_weights_test_driver
    test_model_weights.cpp
)

add_executable(weight_population_test_driver
    test_weight_population.cpp
)

add_executable(tiny_llama_api_test_driver
    test_tiny_llama_api.cpp
)

add_executable(tokenizer_api_test_driver
    test_tokenizer_api.cpp
)

add_executable(api_error_handling_test_driver
    api_error_handling_test_driver.cpp
)

target_link_libraries(vocabulary_test_driver tiny_llama_cpp)
target_link_libraries(tokenizer_test_driver tiny_llama_cpp)
target_link_libraries(ffn_test_driver tiny_llama_cpp)
target_link_libraries(transformer_block_test_driver tiny_llama_cpp)
target_link_libraries(model_weights_test_driver tiny_llama_cpp)
target_link_libraries(weight_population_test_driver tiny_llama_cpp)
target_link_libraries(tiny_llama_api_test_driver tiny_llama_cpp)
target_link_libraries(tokenizer_api_test_driver tiny_llama_cpp)
target_link_libraries(api_error_handling_test_driver tiny_llama_cpp)

set_target_properties(vocabulary_test_driver tokenizer_test_driver ffn_test_driver transformer_block_test_driver model_weights_test_driver weight_population_test_driver tiny_llama_api_test_driver tokenizer_api_test_driver api_error_handling_test_driver
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Add test targets
enable_testing()
add_test(NAME TinyLlamaTests COMMAND run_tests)
add_test(NAME VocabularyTests COMMAND vocabulary_test_driver)
add_test(NAME TokenizerTests COMMAND tokenizer_test_driver)
add_test(NAME FFNTests COMMAND ffn_test_driver)
add_test(NAME TransformerBlockTests COMMAND transformer_block_test_driver)
add_test(NAME ModelWeightsTests COMMAND model_weights_test_driver)
add_test(NAME WeightPopulationTests COMMAND weight_population_test_driver)
add_test(NAME TinyLlamaAPITests COMMAND tiny_llama_api_test_driver)
add_test(NAME TokenizerAPITests COMMAND tokenizer_api_test_driver)
add_test(NAME APIErrorHandlingTests COMMAND api_error_handling_test_driver)