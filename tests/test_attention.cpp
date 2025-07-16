#include "tiny_llama/model.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace tiny_llama;

// Helper function to check if two floats are approximately equal
bool float_eq(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

// Helper function to check if two matrices are approximately equal
bool matrix_eq(const Matrix<float>& a, const Matrix<float>& b, float epsilon = 1e-5f) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }
    
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            if (!float_eq(a(i, j), b(i, j), epsilon)) {
                return false;
            }
        }
    }
    
    return true;
}

// Test MultiHeadAttention initialization
void test_attention_init() {
    std::cout << "Testing attention initialization..." << std::endl;
    
    // Test valid initialization
    MultiHeadAttention attention(512, 8);
    assert(attention.get_model_dim() == 512);
    assert(attention.get_num_heads() == 8);
    
    // Test invalid initialization (model_dim not divisible by num_heads)
    bool exception_thrown = false;
    try {
        MultiHeadAttention invalid_attention(510, 8);
    } catch (const ConfigurationException& e) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Attention initialization tests passed!" << std::endl;
}

// Test wrapper class for accessing private methods
class TestAttention : public MultiHeadAttention {
public:
    TestAttention(int model_dim, int num_heads) : MultiHeadAttention(model_dim, num_heads) {}
    
    Matrix<float> test_scaled_dot_product(
        const Matrix<float>& Q, 
        const Matrix<float>& K, 
        const Matrix<float>& V,
        const Matrix<float>* mask = nullptr) const {
        return scaled_dot_product_attention(Q, K, V, mask);
    }
};

// Test scaled dot-product attention
void test_scaled_dot_product_attention() {
    std::cout << "Testing scaled dot-product attention..." << std::endl;
    
    // Create a simple test case
    Matrix<float> Q(2, 4);
    Matrix<float> K(2, 4);
    Matrix<float> V(2, 4);
    
    // Initialize with known values
    // Q = [[1, 0, 0, 0], [0, 1, 0, 0]]
    Q(0, 0) = 1.0f; Q(0, 1) = 0.0f; Q(0, 2) = 0.0f; Q(0, 3) = 0.0f;
    Q(1, 0) = 0.0f; Q(1, 1) = 1.0f; Q(1, 2) = 0.0f; Q(1, 3) = 0.0f;
    
    // K = [[1, 0, 0, 0], [0, 1, 0, 0]]
    K(0, 0) = 1.0f; K(0, 1) = 0.0f; K(0, 2) = 0.0f; K(0, 3) = 0.0f;
    K(1, 0) = 0.0f; K(1, 1) = 1.0f; K(1, 2) = 0.0f; K(1, 3) = 0.0f;
    
    // V = [[1, 2, 3, 4], [5, 6, 7, 8]]
    V(0, 0) = 1.0f; V(0, 1) = 2.0f; V(0, 2) = 3.0f; V(0, 3) = 4.0f;
    V(1, 0) = 5.0f; V(1, 1) = 6.0f; V(1, 2) = 7.0f; V(1, 3) = 8.0f;
    
    // Create attention instance
    MultiHeadAttention attention(4, 1);
    
    // Use the test wrapper
    TestAttention test_attention(4, 1);
    
    // Compute attention
    Matrix<float> output = test_attention.test_scaled_dot_product(Q, K, V);
    
    // Expected output:
    // For Q[0] = [1,0,0,0], attention weights should focus on K[0]
    // For Q[1] = [0,1,0,0], attention weights should focus on K[1]
    // So output should be approximately:
    // [[1, 2, 3, 4], [5, 6, 7, 8]]
    
    assert(output.rows() == 2);
    assert(output.cols() == 4);
    
    // Check if output is close to expected values
    // The first row should be close to V[0]
    assert(float_eq(output(0, 0), 1.0f, 0.1f));
    assert(float_eq(output(0, 1), 2.0f, 0.1f));
    assert(float_eq(output(0, 2), 3.0f, 0.1f));
    assert(float_eq(output(0, 3), 4.0f, 0.1f));
    
    // The second row should be close to V[1]
    assert(float_eq(output(1, 0), 5.0f, 0.1f));
    assert(float_eq(output(1, 1), 6.0f, 0.1f));
    assert(float_eq(output(1, 2), 7.0f, 0.1f));
    assert(float_eq(output(1, 3), 8.0f, 0.1f));
    
    std::cout << "Scaled dot-product attention tests passed!" << std::endl;
}

// Test attention masking (causal attention)
void test_attention_mask() {
    std::cout << "Testing attention masking..." << std::endl;
    
    // Create a simple test case
    Matrix<float> Q(3, 4);
    Matrix<float> K(3, 4);
    Matrix<float> V(3, 4);
    
    // Initialize with simple values
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            Q(i, j) = 1.0f;
            K(i, j) = 1.0f;
            V(i, j) = static_cast<float>(i * 4 + j);
        }
    }
    
    // Create a causal mask (lower triangular)
    Matrix<float> mask(3, 3);
    mask(0, 0) = 1.0f; mask(0, 1) = 0.0f; mask(0, 2) = 0.0f;
    mask(1, 0) = 1.0f; mask(1, 1) = 1.0f; mask(1, 2) = 0.0f;
    mask(2, 0) = 1.0f; mask(2, 1) =1.0f; mask(2, 2) = 1.0f;
    
    // Use the global TestAttention class
    TestAttention test_attention(4, 1);
    
    // Compute attention with mask
    Matrix<float> output = test_attention.test_scaled_dot_product(Q, K, V, &mask);
    
    // Expected behavior:
    // - First token (position 0) can only attend to position 0
    // - Second token (position 1) can attend to positions 0 and 1
    // - Third token (position 2) can attend to all positions
    
    // Check if output has the right dimensions
    assert(output.rows() == 3);
    assert(output.cols() == 4);
    
    // First row should only see the first row of V
    // Second row should be a weighted average of first and second rows of V
    // Third row should be a weighted average of all rows of V
    
    // Check that the first row is close to V[0]
    for (int j = 0; j < 4; ++j) {
        assert(float_eq(output(0, j), V(0, j), 0.1f));
    }
    
    // Check that the third row is a weighted average of all rows
    float avg_value = 0.0f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            avg_value += V(i, j);
        }
    }
    avg_value /= 12.0f;
    
    // The third row should be closer to the average than to any individual row
    float dist_to_avg = 0.0f;
    for (int j = 0; j < 4; ++j) {
        dist_to_avg += std::abs(output(2, j) - avg_value);
    }
    dist_to_avg /=4.0f;
    
    // The distance to the average should be small
    assert(dist_to_avg < 2.0f);
    
    std::cout << "Attention masking tests passed!" << std::endl;
}

// Test full multi-head attention forward pass
void test_multihead_attention_forward() {
    std::cout << "Testing multi-head attention forward pass..." << std::endl;
    
    // Create a simple input
    Matrix<float> input(3, 8); // seq_len=3, model_dim=8
    
    // Initialize with simple values
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 8; ++j) {
            input(i, j) = static_cast<float>(i * 8 + j) / 24.0f;
        }
    }
    
    // Create attention with 2 heads
    MultiHeadAttention attention(8, 2);
    
    // Create a causal mask
    Matrix<float> mask(3, 3);
    mask(0, 0) = 1.0f; mask(0, 1) = 0.0f; mask(0, 2) = 0.0f;
    mask(1, 0) = 1.0f; mask(1, 1) = 1.0f; mask(1, 2) = 0.0f;
    mask(2, 0) = 1.0f; mask(2, 1) = 1.0f; mask(2, 2) = 1.0f;
    
    // Compute forward pass
    Matrix<float> output = attention.forward(input, &mask);
    
    // Check output dimensions
    assert(output.rows() == 3);
    assert(output.cols() == 8);
    
    // We can't easily predict the exact output values due to random initialization,
    // but we can check that the output is not all zeros or NaNs
    bool all_zeros = true;
    bool has_nan = false;
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (std::abs(output(i, j)) > 1e-6f) {
                all_zeros = false;
            }
            if (std::isnan(output(i, j))) {
                has_nan = true;
            }
        }
    }
    
    assert(!all_zeros);
    assert(!has_nan);
    
    std::cout << "Multi-head attention forward pass tests passed!" << std::endl;
}

// Main test function for attention
void test_attention() {
    std::cout << "\n=== Attention Tests ===" << std::endl;
    
    test_attention_init();
    test_scaled_dot_product_attention();
    test_attention_mask();
    test_multihead_attention_forward();
    
    std::cout << "All attention tests passed!" << std::endl;
}