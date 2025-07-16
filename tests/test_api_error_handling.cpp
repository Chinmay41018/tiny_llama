#include "tiny_llama/tiny_llama.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <limits>
#include <cmath>
#include <unistd.h>
#include <cstdio>

using namespace tiny_llama;

// Test helper functions
void create_test_directory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

void create_test_file(const std::string& path, const std::string& content = "test") {
    std::ofstream file(path);
    file << content;
    file.close();
}

void remove_test_file(const std::string& path) {
    remove(path.c_str());
}

void remove_test_directory(const std::string& path) {
    rmdir(path.c_str());
}

// Test macros
#define EXPECT_EXCEPTION(exception_type, code) \
    do { \
        bool caught = false; \
        try { \
            code; \
        } catch (const exception_type& e) { \
            caught = true; \
            std::cout << "  Expected exception caught: " << e.what() << std::endl; \
        } catch (const std::exception& e) { \
            std::cout << "  Unexpected exception type: " << e.what() << std::endl; \
            assert(false && "Wrong exception type"); \
        } \
        assert(caught && "Expected exception was not thrown"); \
    } while(0)

#define EXPECT_NO_EXCEPTION(code) \
    do { \
        try { \
            code; \
        } catch (const std::exception& e) { \
            std::cout << "  Unexpected exception: " << e.what() << std::endl; \
            assert(false && "Unexpected exception thrown"); \
        } \
    } while(0)

// Test initialization error handling
void test_initialization_errors() {
    std::cout << "Testing initialization error handling..." << std::endl;
    
    TinyLlama llama;
    
    // Test empty model path
    std::cout << "  Testing empty model path..." << std::endl;
    EXPECT_EXCEPTION(FileIOException, llama.initialize(""));
    
    // Test non-existent directory
    std::cout << "  Testing non-existent directory..." << std::endl;
    EXPECT_EXCEPTION(FileIOException, llama.initialize("/non/existent/path"));
    
    // Test file instead of directory
    std::cout << "  Testing file instead of directory..." << std::endl;
    create_test_file("test_file.txt");
    EXPECT_EXCEPTION(FileIOException, llama.initialize("test_file.txt"));
    remove_test_file("test_file.txt");
    
    // Test directory without required files
    std::cout << "  Testing directory without required files..." << std::endl;
    create_test_directory("test_model_dir");
    EXPECT_EXCEPTION(FileIOException, llama.initialize("test_model_dir"));
    remove_test_directory("test_model_dir");
    
    // Test directory with some but not all required files
    std::cout << "  Testing directory with incomplete files..." << std::endl;
    create_test_directory("test_model_dir");
    create_test_file("test_model_dir/vocab.txt");
    EXPECT_EXCEPTION(FileIOException, llama.initialize("test_model_dir"));
    remove_test_file("test_model_dir/vocab.txt");
    remove_test_directory("test_model_dir");
    
    // Test initialize_with_config with invalid files
    std::cout << "  Testing initialize_with_config with invalid files..." << std::endl;
    EXPECT_EXCEPTION(FileIOException, llama.initialize_with_config("", "", ""));
    EXPECT_EXCEPTION(FileIOException, llama.initialize_with_config("nonexistent.txt", "nonexistent.txt", "nonexistent.txt"));
    
    // Test double initialization
    std::cout << "  Testing double initialization..." << std::endl;
    create_test_directory("test_model_dir");
    create_test_file("test_model_dir/vocab.txt", "test_token 0\n");
    create_test_file("test_model_dir/merges.txt", "t e 100\n");
    create_test_file("test_model_dir/weights.bin", "dummy_weights");
    
    // First initialization should work (though it may fail due to invalid format, that's expected)
    try {
        llama.initialize("test_model_dir");
    } catch (const std::exception& e) {
        // Expected to fail due to invalid file format, but that's not what we're testing
        std::cout << "  First initialization failed as expected: " << e.what() << std::endl;
    }
    
    // Second initialization should fail with double initialization error
    EXPECT_EXCEPTION(ModelException, llama.initialize("test_model_dir"));
    
    // Cleanup
    remove_test_file("test_model_dir/vocab.txt");
    remove_test_file("test_model_dir/merges.txt");
    remove_test_file("test_model_dir/weights.bin");
    remove_test_directory("test_model_dir");
    
    std::cout << "  Initialization error handling tests passed!" << std::endl;
}

// Test generation error handling
void test_generation_errors() {
    std::cout << "Testing generation error handling..." << std::endl;
    
    TinyLlama llama;
    
    // Test generation without initialization
    std::cout << "  Testing generation without initialization..." << std::endl;
    EXPECT_EXCEPTION(ModelException, llama.generate("test prompt"));
    
    // Test empty prompt
    std::cout << "  Testing empty prompt..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.generate(""));
    
    // Test invalid max_tokens
    std::cout << "  Testing invalid max_tokens..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.generate("test", 0));
    EXPECT_EXCEPTION(ConfigurationException, llama.generate("test", -1));
    EXPECT_EXCEPTION(ConfigurationException, llama.generate("test", 20000)); // Too large
    
    // Test prompt with null characters
    std::cout << "  Testing prompt with null characters..." << std::endl;
    std::string null_prompt = "test\0prompt";
    EXPECT_EXCEPTION(ConfigurationException, llama.generate(null_prompt));
    
    // Test extremely long prompt
    std::cout << "  Testing extremely long prompt..." << std::endl;
    std::string long_prompt(2000000, 'a'); // 2MB string
    EXPECT_EXCEPTION(ConfigurationException, llama.generate(long_prompt));
    
    std::cout << "  Generation error handling tests passed!" << std::endl;
}

// Test tokenization error handling
void test_tokenization_errors() {
    std::cout << "Testing tokenization error handling..." << std::endl;
    
    TinyLlama llama;
    
    // Test tokenization without initialization
    std::cout << "  Testing tokenization without initialization..." << std::endl;
    EXPECT_EXCEPTION(TokenizerException, llama.tokenize_to_strings("test"));
    EXPECT_EXCEPTION(TokenizerException, llama.tokenize_to_ids("test"));
    
    // Test detokenization without initialization
    std::cout << "  Testing detokenization without initialization..." << std::endl;
    std::vector<int> tokens = {1, 2, 3};
    EXPECT_EXCEPTION(TokenizerException, llama.detokenize(tokens));
    
    // Test text with null characters
    std::cout << "  Testing text with null characters..." << std::endl;
    std::string null_text = "test\0text";
    EXPECT_EXCEPTION(ConfigurationException, llama.tokenize_to_strings(null_text));
    EXPECT_EXCEPTION(ConfigurationException, llama.tokenize_to_ids(null_text));
    
    // Test extremely long text
    std::cout << "  Testing extremely long text..." << std::endl;
    std::string long_text(2000000, 'a'); // 2MB string
    EXPECT_EXCEPTION(ConfigurationException, llama.tokenize_to_strings(long_text));
    EXPECT_EXCEPTION(ConfigurationException, llama.tokenize_to_ids(long_text));
    
    // Test detokenization with invalid token IDs
    std::cout << "  Testing detokenization with invalid token IDs..." << std::endl;
    std::vector<int> negative_tokens = {1, -1, 3};
    EXPECT_EXCEPTION(ConfigurationException, llama.detokenize(negative_tokens));
    
    // Test detokenization with too many tokens
    std::cout << "  Testing detokenization with too many tokens..." << std::endl;
    std::vector<int> too_many_tokens(200000, 1); // 200k tokens
    EXPECT_EXCEPTION(ConfigurationException, llama.detokenize(too_many_tokens));
    
    // Test that empty inputs are allowed for tokenization
    std::cout << "  Testing empty inputs (should be allowed)..." << std::endl;
    EXPECT_NO_EXCEPTION(llama.tokenize_to_strings(""));
    EXPECT_NO_EXCEPTION(llama.tokenize_to_ids(""));
    
    // Test that empty token vector is allowed for detokenization
    std::cout << "  Testing empty token vector (should be allowed)..." << std::endl;
    std::vector<int> empty_tokens;
    EXPECT_NO_EXCEPTION(llama.detokenize(empty_tokens));
    
    std::cout << "  Tokenization error handling tests passed!" << std::endl;
}

// Test configuration error handling
void test_configuration_errors() {
    std::cout << "Testing configuration error handling..." << std::endl;
    
    TinyLlama llama;
    
    // Test temperature validation
    std::cout << "  Testing temperature validation..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(0.0f));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(-1.0f));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(0.005f)); // Below minimum
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(2000.0f)); // Too large
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(std::numeric_limits<float>::infinity()));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(std::numeric_limits<float>::quiet_NaN()));
    
    // Test valid temperature values
    std::cout << "  Testing valid temperature values..." << std::endl;
    EXPECT_NO_EXCEPTION(llama.set_temperature(0.1f));
    EXPECT_NO_EXCEPTION(llama.set_temperature(1.0f));
    EXPECT_NO_EXCEPTION(llama.set_temperature(2.0f));
    
    // Test max sequence length validation
    std::cout << "  Testing max sequence length validation..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(0));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(-1));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(200000)); // Too large
    
    // Test that runtime sequence length change is not supported
    std::cout << "  Testing runtime sequence length change (should fail)..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(1024));
    
    std::cout << "  Configuration error handling tests passed!" << std::endl;
}

// Test status checking and validation
void test_status_validation() {
    std::cout << "Testing status validation..." << std::endl;
    
    TinyLlama llama;
    
    // Test get_vocab_size without initialization
    std::cout << "  Testing get_vocab_size without initialization..." << std::endl;
    EXPECT_EXCEPTION(ModelException, llama.get_vocab_size());
    
    // Test is_ready status
    std::cout << "  Testing is_ready status..." << std::endl;
    assert(!llama.is_ready() && "Model should not be ready before initialization");
    
    std::cout << "  Status validation tests passed!" << std::endl;
}

// Test edge cases and boundary conditions
void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    TinyLlama llama;
    
    // Test boundary values for integers
    std::cout << "  Testing boundary values..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.generate("test", 1000001)); // Just over limit
    EXPECT_NO_EXCEPTION(llama.generate("test", 10000)); // At limit (will fail due to no init, but validation should pass)
    
    // Test boundary values for temperature
    std::cout << "  Testing temperature boundary values..." << std::endl;
    EXPECT_NO_EXCEPTION(llama.set_temperature(0.01f)); // At minimum
    EXPECT_NO_EXCEPTION(llama.set_temperature(1000.0f)); // At maximum
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(1000.1f)); // Just over limit
    
    std::cout << "  Edge case tests passed!" << std::endl;
}

// Test exception context and information
void test_exception_context() {
    std::cout << "Testing exception context and information..." << std::endl;
    
    TinyLlama llama;
    
    // Test that exceptions contain proper context information
    std::cout << "  Testing exception context information..." << std::endl;
    try {
        llama.initialize("");
        assert(false && "Should have thrown exception");
    } catch (const TinyLlamaException& e) {
        std::string what_str = e.what();
        std::cout << "  Exception message: " << what_str << std::endl;
        
        // Check that the exception contains context information
        assert(!e.message().empty() && "Exception should have a message");
        assert(!e.context().empty() && "Exception should have context");
        assert(!e.file().empty() && "Exception should have file information");
        assert(e.line() > 0 && "Exception should have line information");
        
        std::cout << "  Message: " << e.message() << std::endl;
        std::cout << "  Context: " << e.context() << std::endl;
        std::cout << "  File: " << e.file() << std::endl;
        std::cout << "  Line: " << e.line() << std::endl;
    }
    
    std::cout << "  Exception context tests passed!" << std::endl;
}

int main() {
    std::cout << "Running API Error Handling Tests..." << std::endl;
    
    try {
        test_initialization_errors();
        test_generation_errors();
        test_tokenization_errors();
        test_configuration_errors();
        test_status_validation();
        test_edge_cases();
        test_exception_context();
        
        std::cout << "\nAll API error handling tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}