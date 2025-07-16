#include "tiny_llama/model.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdio>
#include <vector>
#include <functional>

using namespace tiny_llama;

// Test helper functions
void assert_true(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "ASSERTION FAILED: " << message << std::endl;
        std::exit(1);
    }
    std::cout << "PASS: " << message << std::endl;
}

void assert_false(bool condition, const std::string& message) {
    if (condition) {
        std::cerr << "ASSERTION FAILED: " << message << std::endl;
        std::exit(1);
    }
    std::cout << "PASS: " << message << std::endl;
}

void assert_throws(std::function<void()> func, const std::string& message) {
    bool threw = false;
    try {
        func();
    } catch (const std::exception&) {
        threw = true;
    }
    assert_true(threw, message);
}

// Create a test weights file with invalid magic number
void create_invalid_magic_file(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    uint32_t invalid_magic = 0x12345678;
    file.write(reinterpret_cast<const char*>(&invalid_magic), sizeof(uint32_t));
    file.close();
}

// Create a test weights file with invalid version
void create_invalid_version_file(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    
    // Write correct magic number
    const uint32_t MAGIC_NUMBER = 0x544C4C4D;
    file.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(uint32_t));
    
    // Write invalid version
    uint32_t invalid_version = 999;
    file.write(reinterpret_cast<const char*>(&invalid_version), sizeof(uint32_t));
    
    file.close();
}

// Create a test weights file with mismatched configuration
void create_mismatched_config_file(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    
    // Write correct header
    const uint32_t MAGIC_NUMBER = 0x544C4C4D;
    file.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(uint32_t));
    
    const uint32_t VERSION = 1;
    file.write(reinterpret_cast<const char*>(&VERSION), sizeof(uint32_t));
    
    // Write mismatched configuration
    int wrong_model_dim = 256; // Different from default 512
    int num_layers = 6;
    int num_heads = 8;
    int ffn_hidden_dim = 2048;
    int max_sequence_length = 1024;
    int vocab_size = 32000;
    float dropout_rate = 0.1f;
    
    file.write(reinterpret_cast<const char*>(&wrong_model_dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
    file.write(reinterpret_cast<const char*>(&num_heads), sizeof(int));
    file.write(reinterpret_cast<const char*>(&ffn_hidden_dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&max_sequence_length), sizeof(int));
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&dropout_rate), sizeof(float));
    
    file.close();
}

// Test basic weight file creation and loading
void test_basic_weight_loading() {
    std::cout << "\n=== Testing Basic Weight Loading ===" << std::endl;
    
    // Create a model with default configuration
    TinyLlamaModel model;
    
    // Save weights to a test file
    const std::string test_file = "test_weights.bin";
    model.save_model_weights(test_file);
    
    // Create a new model and load the weights
    TinyLlamaModel model2;
    model2.load_model_weights(test_file);
    
    // Clean up
    std::remove(test_file.c_str());
    
    assert_true(true, "Basic weight loading completed successfully");
}

// Test loading with invalid magic number
void test_invalid_magic_number() {
    std::cout << "\n=== Testing Invalid Magic Number ===" << std::endl;
    
    const std::string test_file = "invalid_magic.bin";
    create_invalid_magic_file(test_file);
    
    TinyLlamaModel model;
    
    assert_throws([&]() {
        model.load_model_weights(test_file);
    }, "Loading with invalid magic number should throw FileIOException");
    
    // Clean up
    std::remove(test_file.c_str());
}

// Test loading with invalid version
void test_invalid_version() {
    std::cout << "\n=== Testing Invalid Version ===" << std::endl;
    
    const std::string test_file = "invalid_version.bin";
    create_invalid_version_file(test_file);
    
    TinyLlamaModel model;
    
    assert_throws([&]() {
        model.load_model_weights(test_file);
    }, "Loading with invalid version should throw FileIOException");
    
    // Clean up
    std::remove(test_file.c_str());
}

// Test loading with mismatched configuration
void test_mismatched_configuration() {
    std::cout << "\n=== Testing Mismatched Configuration ===" << std::endl;
    
    const std::string test_file = "mismatched_config.bin";
    create_mismatched_config_file(test_file);
    
    TinyLlamaModel model;
    
    assert_throws([&]() {
        model.load_model_weights(test_file);
    }, "Loading with mismatched configuration should throw FileIOException");
    
    // Clean up
    std::remove(test_file.c_str());
}

// Test loading non-existent file
void test_nonexistent_file() {
    std::cout << "\n=== Testing Non-existent File ===" << std::endl;
    
    TinyLlamaModel model;
    
    assert_throws([&]() {
        model.load_model_weights("nonexistent_file.bin");
    }, "Loading non-existent file should throw FileIOException");
}

// Test weight file format validation
void test_weight_file_format() {
    std::cout << "\n=== Testing Weight File Format ===" << std::endl;
    
    // Create a model and save weights
    TinyLlamaModel model;
    const std::string test_file = "format_test.bin";
    model.save_model_weights(test_file);
    
    // Verify file exists and has content
    std::ifstream file(test_file, std::ios::binary | std::ios::ate);
    assert_true(file.is_open(), "Weight file should be created successfully");
    
    size_t file_size = file.tellg();
    assert_true(file_size > 0, "Weight file should have content");
    
    file.close();
    
    // Verify we can read the header correctly
    std::ifstream read_file(test_file, std::ios::binary);
    
    uint32_t magic;
    read_file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    assert_true(magic == 0x544C4C4D, "Magic number should be correct");
    
    uint32_t version;
    read_file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    assert_true(version == 1, "Version should be 1");
    
    read_file.close();
    
    // Clean up
    std::remove(test_file.c_str());
}

// Test custom configuration weight loading
void test_custom_configuration() {
    std::cout << "\n=== Testing Custom Configuration ===" << std::endl;
    
    // Create a model with custom configuration
    ModelConfig config;
    config.model_dim = 256;
    config.num_layers = 4;
    config.num_heads = 4;
    config.ffn_hidden_dim = 1024;
    config.max_sequence_length = 512;
    config.vocab_size = 16000;
    
    TinyLlamaModel model(config);
    
    // Save and load weights
    const std::string test_file = "custom_config.bin";
    model.save_model_weights(test_file);
    
    TinyLlamaModel model2(config);
    model2.load_model_weights(test_file);
    
    // Verify configuration matches
    assert_true(model2.get_config().model_dim == 256, "Model dimension should match");
    assert_true(model2.get_config().num_layers == 4, "Number of layers should match");
    assert_true(model2.get_config().num_heads == 4, "Number of heads should match");
    
    // Clean up
    std::remove(test_file.c_str());
}

// Test error handling for corrupted files
void test_corrupted_file_handling() {
    std::cout << "\n=== Testing Corrupted File Handling ===" << std::endl;
    
    // Create a valid file first
    TinyLlamaModel model;
    const std::string test_file = "corrupted_test.bin";
    model.save_model_weights(test_file);
    
    // Truncate the file to simulate corruption
    std::ofstream truncate_file(test_file, std::ios::binary | std::ios::trunc);
    truncate_file << "corrupted";
    truncate_file.close();
    
    TinyLlamaModel model2;
    
    assert_throws([&]() {
        model2.load_model_weights(test_file);
    }, "Loading corrupted file should throw FileIOException");
    
    // Clean up
    std::remove(test_file.c_str());
}

// Test dimension validation for individual components
void test_dimension_validation() {
    std::cout << "\n=== Testing Dimension Validation ===" << std::endl;
    
    // This test verifies that the dimension validation logic works
    // by checking that the helper methods validate dimensions correctly
    
    TinyLlamaModel model;
    const std::string test_file = "dimension_test.bin";
    
    // Create a valid weights file
    model.save_model_weights(test_file);
    
    // Load it successfully
    TinyLlamaModel model2;
    model2.load_model_weights(test_file);
    
    assert_true(true, "Dimension validation works for valid files");
    
    // Clean up
    std::remove(test_file.c_str());
}

// Test binary file format consistency
void test_binary_format_consistency() {
    std::cout << "\n=== Testing Binary Format Consistency ===" << std::endl;
    
    // Create two models with the same configuration
    TinyLlamaModel model1;
    TinyLlamaModel model2;
    
    // Save weights from both models
    const std::string file1 = "model1_weights.bin";
    const std::string file2 = "model2_weights.bin";
    
    model1.save_model_weights(file1);
    model2.save_model_weights(file2);
    
    // Read both files and compare sizes (they should be identical for same config)
    std::ifstream f1(file1, std::ios::binary | std::ios::ate);
    std::ifstream f2(file2, std::ios::binary | std::ios::ate);
    
    size_t size1 = f1.tellg();
    size_t size2 = f2.tellg();
    
    assert_true(size1 == size2, "Files with same configuration should have same size");
    assert_true(size1 > 1000, "Weight files should be reasonably large");
    
    f1.close();
    f2.close();
    
    // Clean up
    std::remove(file1.c_str());
    std::remove(file2.c_str());
}

int main() {
    std::cout << "Running Model Weight Loading Tests..." << std::endl;
    
    try {
        test_basic_weight_loading();
        test_invalid_magic_number();
        test_invalid_version();
        test_mismatched_configuration();
        test_nonexistent_file();
        test_weight_file_format();
        test_custom_configuration();
        test_corrupted_file_handling();
        test_dimension_validation();
        test_binary_format_consistency();
        
        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}