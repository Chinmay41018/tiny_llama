#include "tiny_llama/model.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <fstream>

using namespace tiny_llama;

// Helper function to check if two floats are approximately equal
bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// Helper function to calculate GELU activation for testing
std::vector<float> calculate_gelu(const std::vector<float>& input) {
    const float sqrt_2_over_pi =0.7978845608028654f;  // sqrt(2/π)
    const float coeff = 0.044715f;
    
    std::vector<float> result(input.size());
    
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        float tanh_inner = std::tanh(inner);
        result[i] = 0.5f * x * (1.0f + tanh_inner);
    }
    
    return result;
}

// Test GELU activation function indirectly through forward pass
void test_gelu_activation() {
    std::cout << "Testing GELU activation function..." << std::endl;
    
    // Create a simple FFN with small dimensions for testing
    int model_dim = 1;
    int hidden_dim = 5;
    FeedForwardNetwork ffn(model_dim, hidden_dim);
    
    // Create a test input with known values
    Matrix<float> input(1, model_dim);
    input(0, 0) = 1.0f;
    
    // Create weights that will pass through our test values
    Matrix<float> linear1_weights(model_dim, hidden_dim);
    linear1_weights(0, 0) = -2.0f;
    linear1_weights(0, 1) = -1.0f;
    linear1_weights(0, 2) = 0.0f;
    linear1_weights(0, 3) = 1.0f;
    linear1_weights(0, 4) = 2.0f;
    
    // Create a temporary file with test weights
    std::string temp_file = "test_gelu_weights.bin";
    std::ofstream file(temp_file, std::ios::binary);
    assert(file.is_open());
    
    // Write linear1 weights
    size_t rows = linear1_weights.rows();
    size_t cols = linear1_weights.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear1_weights.data()), rows * cols * sizeof(float));
    
    // Write linear1 bias (all zeros)
    size_t bias1_size = hidden_dim;
    std::vector<float> linear1_bias(hidden_dim, 0.0f);
    file.write(reinterpret_cast<const char*>(&bias1_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear1_bias.data()), bias1_size * sizeof(float));
    
    // Write linear2 weights (identity matrix)
    Matrix<float> linear2_weights(hidden_dim, model_dim);
    for (size_t i = 0; i < hidden_dim; ++i) {
        linear2_weights(i, 0) = 1.0f;
    }
    rows = linear2_weights.rows();
    cols = linear2_weights.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear2_weights.data()), rows * cols * sizeof(float));
    
    // Write linear2 bias (all zeros)
    size_t bias2_size = model_dim;
    std::vector<float> linear2_bias(model_dim, 0.0f);
    file.write(reinterpret_cast<const char*>(&bias2_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear2_bias.data()), bias2_size * sizeof(float));
    
    file.close();
    
    // Load the weights
    ffn.load_weights(temp_file);
    
    // Run forward pass
    Matrix<float> output = ffn.forward(input);
    
    // Expected values after GELU
    std::vector<float> test_values = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> expected_gelu = calculate_gelu(test_values);
    
    // The output should be the sum of the GELU activations
    float expected_sum = 0.0f;
    for (float val : expected_gelu) {
        expected_sum += val;
    }
    
    if (!approx_equal(output(0, 0), expected_sum)) {
        std::cerr << "GELU test failed: expected " << expected_sum 
                  << ", got " << output(0, 0) << std::endl;
        assert(false);
    }
    
    // Clean up
    std::remove(temp_file.c_str());
    
    std::cout << "GELU activation test passed!" << std::endl;
}

// Test forward pass with simple input
void test_forward_pass() {
    std::cout << "Testing FFN forward pass..." << std::endl;
    
    // Create a simple FFN with small dimensions for testing
    int model_dim = 3;
    int hidden_dim = 4;
    FeedForwardNetwork ffn(model_dim, hidden_dim);
    
    // Set weights and biases manually for predictable output
    Matrix<float> linear1_weights(model_dim, hidden_dim);
    for (size_t i = 0; i < model_dim; ++i) {
        for (size_t j = 0; j < hidden_dim; ++j) {
            linear1_weights(i, j) = 0.1f * (i + 1) * (j + 1);
        }
    }
    
    std::vector<float> linear1_bias = {0.1f, 0.2f, 0.3f, 0.4f};
    
    Matrix<float> linear2_weights(hidden_dim, model_dim);
    for (size_t i = 0; i < hidden_dim; ++i) {
        for (size_t j = 0; j < model_dim; ++j) {
            linear2_weights(i, j) = 0.05f * (i + 1) * (j + 1);
        }
    }
    
    std::vector<float> linear2_bias = {0.01f, 0.02f, 0.03f};
    
    // Create a temporary file with test weights
    std::string temp_file = "test_ffn_forward_weights.bin";
    std::ofstream file(temp_file, std::ios::binary);
    assert(file.is_open());
    
    // Write linear1 weights
    size_t rows = linear1_weights.rows();
    size_t cols = linear1_weights.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear1_weights.data()), rows * cols * sizeof(float));
    
    // Write linear1 bias
    size_t bias1_size = linear1_bias.size();
    file.write(reinterpret_cast<const char*>(&bias1_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear1_bias.data()), bias1_size * sizeof(float));
    
    // Write linear2 weights
    rows = linear2_weights.rows();
    cols = linear2_weights.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear2_weights.data()), rows * cols * sizeof(float));
    
    // Write linear2 bias
    size_t bias2_size = linear2_bias.size();
    file.write(reinterpret_cast<const char*>(&bias2_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear2_bias.data()), bias2_size * sizeof(float));
    
    file.close();
    
    // Load the weights
    ffn.load_weights(temp_file);
    
    // Create a simple input
    Matrix<float> input(2, model_dim);
    input(0,0) = 1.0f; input(0, 1) = 2.0f; input(0, 2) = 3.0f;
    input(1, 0) = 4.0f; input(1, 1) = 5.0f; input(1, 2) = 6.0f;
    
    // Compute expected output manually
    // First linear layer: input * linear1_weights + bias
    Matrix<float> expected_hidden(2, hidden_dim);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < hidden_dim; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < model_dim; ++k) {
                sum += input(i, k) * linear1_weights(k, j);
            }
            expected_hidden(i, j) = sum + linear1_bias[j];
        }
    }
    
    // Apply GELU activation
    for (size_t i = 0; i < 2; ++i) {
        std::vector<float> row_data(hidden_dim);
        for (size_t j = 0; j < hidden_dim; ++j) {
            row_data[j] = expected_hidden(i, j);
        }
        
        std::vector<float> activated = calculate_gelu(row_data);
        
        for (size_t j = 0; j < hidden_dim; ++j) {
            expected_hidden(i, j) = activated[j];
        }
    }
    
    // Second linear layer: hidden * linear2_weights + bias
    Matrix<float> expected_output(2, model_dim);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < model_dim; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < hidden_dim; ++k) {
                sum += expected_hidden(i, k) * linear2_weights(k, j);
            }
            expected_output(i, j) = sum + linear2_bias[j];
        }
    }
    
    // Compute actual output
    Matrix<float> output = ffn.forward(input);
    
    // Compare expected and actual outputs
    assert(output.rows() == expected_output.rows());
    assert(output.cols() == expected_output.cols());
    
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            if (!approx_equal(output(i, j), expected_output(i, j))) {
                std::cerr << "Forward pass test failed at position (" << i << ", " << j << "): " 
                          << "expected " << expected_output(i, j) << ", got " << output(i, j) << std::endl;
                assert(false);
            }
        }
    }
    
    // Clean up
    std::remove(temp_file.c_str());
    
    std::cout << "Forward pass test passed!" << std::endl;
}

// Test dimension mismatch error handling
void test_dimension_mismatch() {
    std::cout << "Testing dimension mismatch handling..." << std::endl;
    
    FeedForwardNetwork ffn(512, 2048);
    
    // Create input with wrong dimensions
    Matrix<float> input(10, 256);  // Should be 512 columns
    
    try {
        Matrix<float> output = ffn.forward(input);
        std::cerr << "Expected exception for dimension mismatch, but none was thrown!" << std::endl;
        assert(false);
    } catch (const ModelException& e) {
        // Expected exception
        std::cout << "Correctly caught exception: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Caught unexpected exception: " << e.what() << std::endl;
        assert(false);
    }
    
    std::cout << "Dimension mismatch test passed!" << std::endl;
}

// Test weight loading functionality
void test_weight_loading() {
    std::cout << "Testing weight loading..." << std::endl;
    
    // Create a temporary file with test weights
    std::string temp_file = "test_ffn_weights.bin";
    
    // Create test data
    int model_dim = 3;
    int hidden_dim = 4;
    
    Matrix<float> linear1_weights(model_dim, hidden_dim);
    for (size_t i = 0; i < model_dim; ++i) {
        for (size_t j = 0; j < hidden_dim; ++j) {
            linear1_weights(i, j) = 0.1f * (i + 1) * (j + 1);
        }
    }
    
    std::vector<float> linear1_bias = {0.1f, 0.2f, 0.3f, 0.4f};
    
    Matrix<float> linear2_weights(hidden_dim, model_dim);
    for (size_t i = 0; i < hidden_dim; ++i) {
        for (size_t j = 0; j < model_dim; ++j) {
            linear2_weights(i, j) = 0.05f * (i + 1) * (j + 1);
        }
    }
    
    std::vector<float> linear2_bias = {0.01f, 0.02f, 0.03f};
    
    // Write test data to file
    std::ofstream file(temp_file, std::ios::binary);
    assert(file.is_open());
    
    // Write linear1 weights
    size_t rows = linear1_weights.rows();
    size_t cols = linear1_weights.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear1_weights.data()), rows * cols * sizeof(float));
    
    // Write linear1 bias
    size_t bias1_size = linear1_bias.size();
    file.write(reinterpret_cast<const char*>(&bias1_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear1_bias.data()), bias1_size * sizeof(float));
    
    // Write linear2 weights
    rows = linear2_weights.rows();
    cols = linear2_weights.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear2_weights.data()), rows * cols * sizeof(float));
    
    // Write linear2 bias
    size_t bias2_size = linear2_bias.size();
    file.write(reinterpret_cast<const char*>(&bias2_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(linear2_bias.data()), bias2_size * sizeof(float));
    
    file.close();
    
    // Create FFN and load weights
    FeedForwardNetwork ffn(model_dim, hidden_dim);
    
    try {
        ffn.load_weights(temp_file);
        
        // Test the loaded weights by running a forward pass
        Matrix<float> input(1, model_dim);
        input(0, 0) = 1.0f;
        input(0, 1) = 1.0f;
        input(0, 2) = 1.0f;
        
        Matrix<float> output = ffn.forward(input);
        
        // We can't directly check the weights, but we can verify the output
        // is consistent with the weights we loaded
        
        // Compute expected output manually
        // First linear layer: input * linear1_weights + bias
        Matrix<float> expected_hidden(1, hidden_dim);
        for (size_t j = 0; j < hidden_dim; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < model_dim; ++k) {
                sum += input(0, k) * linear1_weights(k, j);
            }
            expected_hidden(0, j) = sum + linear1_bias[j];
        }
        
        // Apply GELU activation
        std::vector<float> row_data(hidden_dim);
        for (size_t j = 0; j < hidden_dim; ++j) {
            row_data[j] = expected_hidden(0, j);
        }
        
        std::vector<float> activated = calculate_gelu(row_data);
        
        for (size_t j = 0; j < hidden_dim; ++j) {
            expected_hidden(0, j) = activated[j];
        }
        
        // Second linear layer: hidden * linear2_weights + bias
        Matrix<float> expected_output(1, model_dim);
        for (size_t j = 0; j < model_dim; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < hidden_dim; ++k) {
                sum += expected_hidden(0, k) * linear2_weights(k, j);
            }
            expected_output(0, j) = sum + linear2_bias[j];
        }
        
        // Compare expected and actual outputs
        for (size_t j = 0; j < model_dim; ++j) {
            if (!approx_equal(output(0, j), expected_output(0, j))) {
                std::cerr << "Weight loading test failed at position " << j << ": " 
                          << "expected " << expected_output(0, j) << ", got " << output(0, j) << std::endl;
                assert(false);
            }
        }
        
        std::cout << "Weight loading test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Weight loading test failed: " << e.what() << std::endl;
        assert(false);
    }
    
    // Clean up
    std::remove(temp_file.c_str());
}

// Test functions are called from ffn_test_driver.cpp