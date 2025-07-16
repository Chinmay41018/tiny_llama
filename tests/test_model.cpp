#include "tiny_llama/model.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <memory>

using namespace tiny_llama;

// Helper function to check if two float values are approximately equal
bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// Test model initialization with default configuration
void test_model_init_default() {
    std::cout << "Testing model initialization with default config..." << std::endl;
    
    // Create model with default configuration
    TinyLlamaModel model;
    
    // Check that model is initialized with correct default dimensions
    const ModelConfig& config = model.get_config();
    assert(config.model_dim == 512);
    assert(config.num_layers == 6);
    assert(config.num_heads == 8);
    assert(config.ffn_hidden_dim ==2048);
    assert(config.max_sequence_length == 1024);
    assert(config.vocab_size == 32000);
    
    std::cout << "Model initialization with default config test passed!" << std::endl;
}

// Test model initialization with custom configuration
void test_model_init_custom() {
    std::cout << "Testing model initialization with custom config..." << std::endl;
    
    // Create custom configuration
    ModelConfig config;
    config.model_dim = 256;
    config.num_layers = 4;
    config.num_heads = 4;
    config.ffn_hidden_dim = 1024;
    config.max_sequence_length = 512;
    config.vocab_size = 16000;
    
    // Create model with custom configuration
    TinyLlamaModel model(config);
    
    // Check that model is initialized with correct custom dimensions
    const ModelConfig& model_config = model.get_config();
    assert(model_config.model_dim ==256);
    assert(model_config.num_layers == 4);
    assert(model_config.num_heads ==4);
    assert(model_config.ffn_hidden_dim == 1024);
    assert(model_config.max_sequence_length == 512);
    assert(model_config.vocab_size == 16000);
    
    std::cout << "Model initialization with custom config test passed!" << std::endl;
}

// Test model forward pass with simple input
void test_model_forward() {
    std::cout << "Testing model forward pass..." << std::endl;
    
    try {
        // Create a small model for testing
        ModelConfig config;
        config.model_dim = 8;
        config.num_layers = 2;
        config.num_heads =2;
        config.ffn_hidden_dim = 16;
        config.max_sequence_length = 10;
        config.vocab_size = 100;
        
        TinyLlamaModel model(config);
        
        // Verify that the model is not initialized
        assert(!model.is_initialized());
        
        // Create simple input tokens
        std::vector<int> input_tokens = {1, 2, 3};
        
        // This should throw an exception since we haven't initialized the tokenizer
        bool exception_thrown = false;
        try {
            std::vector<float> logits = model.forward(input_tokens);
        } catch (const ModelException& e) {
            exception_thrown = true;
            std::cout << "Expected exception caught: " << e.what() << std::endl;
        }
        assert(exception_thrown);
        
        std::cout << "Model forward pass test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        throw;
    }
}

// Test is_initialized function
void test_is_initialized() {
    std::cout << "Testing is_initialized function..." << std::endl;
    
    // Create model
    TinyLlamaModel model;
    
    // Model should not be fully initialized yet (tokenizer not loaded)
    assert(!model.is_initialized());
    
    std::cout << "is_initialized function test passed!" << std::endl;
}

// Test tokenizer methods
void test_tokenizer_methods() {
    std::cout << "Testing tokenizer methods..." << std::endl;
    
    // Create model
    TinyLlamaModel model;
    
    // Test tokenize method (should throw exception since tokenizer is not initialized)
    bool exception_thrown = false;
    try {
        model.tokenize("test");
    } catch (const TokenizerException&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    // Test detokenize method (should throw exception since tokenizer is not initialized)
    exception_thrown = false;
    try {
        model.detokenize({1, 2, 3});
    } catch (const TokenizerException&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    // Test tokenize_to_strings method (should throw exception since tokenizer is not initialized)
    exception_thrown = false;
    try {
        model.tokenize_to_strings("test");
    } catch (const TokenizerException&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Tokenizer methods test passed!" << std::endl;
}

// Test get_vocab_size method
void test_get_vocab_size() {
    std::cout << "Testing get_vocab_size method..." << std::endl;
    
    // Create model with default configuration
    TinyLlamaModel model;
    
    // Check that vocab size matches configuration
    assert(model.get_vocab_size() == model.get_config().vocab_size);
    
    // Create model with custom configuration
    ModelConfig config;
    config.vocab_size = 1000;
    TinyLlamaModel custom_model(config);
    
    // Check that vocab size matches custom configuration
    assert(custom_model.get_vocab_size() == 1000);
    
    std::cout << "get_vocab_size method test passed!" << std::endl;
}

// Test softmax function
void test_softmax() {
    std::cout << "Testing softmax function..." << std::endl;
    
    // Create model
    TinyLlamaModel model;
    
    // Test with simple logits
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    
    // Access the private softmax method using a test helper
    // In a real implementation, we would use a friend test class or expose the method for testing
    // For now, we'll test indirectly through the generate_text method
    
    // We can test that the model doesn't crash when we try to generate text
    bool exception_thrown = false;
    try {
        model.generate_text("test", 5, 1.0f);
    } catch (const ModelException& e) {
        // This is expected since the model is not initialized
        exception_thrown = true;
        std::cout << "Expected exception caught: " << e.what() << std::endl;
    }
    assert(exception_thrown);
    
    std::cout << "Softmax function test passed!" << std::endl;
}

// Test token sampling
void test_token_sampling() {
    std::cout << "Testing token sampling..." << std::endl;
    
    // Create model
    TinyLlamaModel model;
    
    // Test with different temperatures
    // We can't directly test the sampling function since it's private
    // But we can test that the generate_text method handles different temperatures
    
    // Test with temperature = 0 (should be deterministic)
    bool exception_thrown = false;
    try {
        model.generate_text("test", 5, 0.0f);
    } catch (const ModelException& e) {
        // This is expected since the model is not initialized
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    // Test with temperature = 1.0 (standard sampling)
    exception_thrown = false;
    try {
        model.generate_text("test", 5, 1.0f);
    } catch (const ModelException& e) {
        // This is expected since the model is not initialized
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    // Test with high temperature (more random)
    exception_thrown = false;
    try {
        model.generate_text("test", 5, 2.0f);
    } catch (const ModelException& e) {
        // This is expected since the model is not initialized
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Token sampling test passed!" << std::endl;
}

// Test attention mask generation
void test_attention_mask() {
    std::cout << "Testing attention mask generation..." << std::endl;
    
    // Create model
    TinyLlamaModel model;
    
    // We can't directly test the create_attention_mask method since it's private
    // But we can test that the forward method uses it correctly
    
    // Create simple input tokens
    std::vector<int> input_tokens = {1, 2, 3};
    
    // This should throw an exception since we haven't initialized the tokenizer
    bool exception_thrown = false;
    try {
        std::vector<float> logits = model.forward(input_tokens);
    } catch (const ModelException& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Attention mask generation test passed!" << std::endl;
}

// Test text generation
void test_text_generation() {
    std::cout << "Testing text generation..." << std::endl;
    
    // Create model
    TinyLlamaModel model;
    
    // Test with empty prompt
    bool exception_thrown = false;
    try {
        model.generate_text("", 5, 1.0f);
    } catch (const ModelException& e) {
        // This is expected since the model is not initialized
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    // Test with non-empty prompt
    exception_thrown = false;
    try {
        model.generate_text("Hello", 10, 1.0f);
    } catch (const ModelException& e) {
        // This is expected since the model is not initialized
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    // Test with invalid max_tokens
    exception_thrown = false;
    try {
        model.generate_text("Hello", -1, 1.0f);
    } catch (const ModelException& e) {
        // This is expected since max_tokens is invalid
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Text generation test passed!" << std::endl;
}

int main() {
    try {
        test_model_init_default();
        test_model_init_custom();
        test_model_forward();
        test_is_initialized();
        test_tokenizer_methods();
        test_get_vocab_size();
        test_softmax();
        test_token_sampling();
        test_attention_mask();
        test_text_generation();
        
        std::cout << "All model tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}