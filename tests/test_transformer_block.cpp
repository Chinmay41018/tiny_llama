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

// Helper function to check if two matrices are approximately equal
bool matrices_approx_equal(const Matrix<float>& a, const Matrix<float>& b, float epsilon = 1e-5f) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }
    
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            if (!approx_equal(a(i, j), b(i, j), epsilon)) {
                return false;
            }
        }
    }
    
    return true;
}

// Test transformer block initialization
void test_transformer_block_init() {
    std::cout << "Testing transformer block initialization..." << std::endl;
    
    // Create a transformer block
    int model_dim = 4;
    int num_heads = 2;
    int ffn_hidden_dim = 8;
    TransformerBlock block(model_dim, num_heads, ffn_hidden_dim);
    
    // Check model dimension
    assert(block.get_model_dim() == model_dim);
    
    std::cout << "Transformer block initialization test passed!" << std::endl;
}

// Test transformer block forward pass
void test_transformer_block_forward() {
    std::cout << "Testing transformer block forward pass..." << std::endl;
    
    // Create a transformer block
    int model_dim = 4;
    int num_heads = 2;
    int ffn_hidden_dim = 8;
    TransformerBlock block(model_dim, num_heads, ffn_hidden_dim);
    
    // Create input matrix
    Matrix<float> input(2, model_dim);
    input(0, 0) = 0.1f; input(0, 1) = 0.2f; input(0, 2) =0.3f; input(0, 3) = 0.4f;
    input(1, 0) = 0.5f; input(1, 1) = 0.6f; input(1, 2) = 0.7f; input(1, 3) = 0.8f;
    
    // Create attention mask (causal mask)
    Matrix<float> mask(2, 2);
    mask(0, 0) = 1.0f; mask(0, 1) = 0.0f;
    mask(1, 0) = 1.0f; mask(1, 1) = 1.0f;
    
    // Call forward method
    Matrix<float> output = block.forward(input, &mask);
    
    // Check dimensions
    assert(output.rows() == input.rows());
    assert(output.cols() == input.cols());
    
    // We can't easily predict the exact output values since they depend on
    // the random initialization of weights, but we can check that the output
    // is different from the input and has reasonable values
    bool different = false;
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            if (!approx_equal(output(i, j), input(i, j))) {
                different = true;
            }
            // Check that values are not NaN or infinity
            assert(!std::isnan(output(i, j)));
            assert(!std::isinf(output(i, j)));
            // Check that values are within a reasonable range
            assert(std::abs(output(i, j)) < 10.0f);
        }
    }
    assert(different);
    
    std::cout << "Transformer block forward pass test passed!" << std::endl;
}

// Test transformer block with residual connections
void test_residual_connections() {
    std::cout << "Testing residual connections..." << std::endl;
    
    // Create a transformer block
    int model_dim = 4;
    int num_heads = 2;
    int ffn_hidden_dim = 8;
    TransformerBlock block(model_dim, num_heads, ffn_hidden_dim);
    
    // Create input matrix with all zeros
    Matrix<float> input(2, model_dim);
    input.fill(0.0f);
    
    // Process input through block
    Matrix<float> output = block.forward(input, nullptr);
    
    // Check that output is different from input (due to residual connections)
    bool different = false;
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            if (!approx_equal(output(i, j), input(i, j))) {
                different = true;
                break;
            }
        }
        if (different) break;
    }
    assert(different);
    
    std::cout << "Residual connections test passed!" << std::endl;
}

// Test transformer block with null mask
void test_null_mask() {
    std::cout << "Testing transformer block with null mask..." << std::endl;
    
    // Create a transformer block
    int model_dim = 4;
    int num_heads = 2;
    int ffn_hidden_dim = 8;
    TransformerBlock block(model_dim, num_heads, ffn_hidden_dim);
    
    // Create input matrix
    Matrix<float> input(2, model_dim);
    input(0, 0) = 0.1f; input(0, 1) = 0.2f; input(0, 2) = 0.3f; input(0, 3) = 0.4f;
    input(1, 0) = 0.5f; input(1, 1) = 0.6f; input(1, 2) = 0.7f; input(1, 3) = 0.8f;
    
    // Call forward method with null mask
    Matrix<float> output = block.forward(input, nullptr);
    
    // Check dimensions
    assert(output.rows() == input.rows());
    assert(output.cols() == input.cols());
    
    // Check that values are not NaN or infinity
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            assert(!std::isnan(output(i, j)));
            assert(!std::isinf(output(i, j)));
        }
    }
    
    std::cout << "Null mask test passed!" << std::endl;
}

int main() {
    try {
        test_transformer_block_init();
        test_transformer_block_forward();
        test_residual_connections();
        test_null_mask();
        
        std::cout << "All transformer block tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}