#include "tiny_llama/model.hpp"
#include "tiny_llama/exceptions.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace tiny_llama {

// MultiHeadAttention implementation
MultiHeadAttention::MultiHeadAttention(int model_dim, int num_heads) 
    : num_heads_(num_heads), model_dim_(model_dim) {
    if (model_dim % num_heads != 0) {
        throw ConfigurationException("Model dimension must be divisible by number of heads");
    }
    head_dim_ = model_dim / num_heads;
    
    // Initialize weight matrices
    query_weights_ = Matrix<float>(model_dim, model_dim);
    key_weights_ = Matrix<float>(model_dim, model_dim);
    value_weights_ = Matrix<float>(model_dim, model_dim);
    output_weights_ = Matrix<float>(model_dim, model_dim);
    
    // Initialize with small random values (Xavier/Glorot initialization)
    float scale = std::sqrt(6.0f / (model_dim + model_dim));
    
    // We'll use a simple random initialization for now
    // In a real implementation, we would use a proper random number generator
    for (size_t i = 0; i < query_weights_.size(); ++i) {
        query_weights_.data()[i] = (static_cast<float>(rand()) / RAND_MAX *2.0f - 1.0f) * scale;
        key_weights_.data()[i] = (static_cast<float>(rand()) / RAND_MAX *2.0f - 1.0f) * scale;
        value_weights_.data()[i] = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * scale;
        output_weights_.data()[i] = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

void MultiHeadAttention::load_weights(const std::string& weights_file) {
    try {
        // Load weights from binary file
        query_weights_.load_from_file(weights_file + ".query");
        key_weights_.load_from_file(weights_file + ".key");
        value_weights_.load_from_file(weights_file + ".value");
        output_weights_.load_from_file(weights_file + ".output");
        
        // Validate dimensions
        if (query_weights_.rows() != model_dim_ || query_weights_.cols() != model_dim_) {
            throw FileIOException("Query weights dimensions mismatch");
        }
        if (key_weights_.rows() != model_dim_ || key_weights_.cols() != model_dim_) {
            throw FileIOException("Key weights dimensions mismatch");
        }
        if (value_weights_.rows() != model_dim_ || value_weights_.cols() != model_dim_) {
            throw FileIOException("Value weights dimensions mismatch");
        }
        if (output_weights_.rows() != model_dim_ || output_weights_.cols() != model_dim_) {
            throw FileIOException("Output weights dimensions mismatch");
        }
    } catch (const std::exception& e) {
        throw FileIOException(std::string("Failed to load attention weights: ") + e.what());
    }
}

Matrix<float> MultiHeadAttention::forward(const Matrix<float>& input, const Matrix<float>* mask) const {
    // Input shape: [batch_size, seq_len, model_dim]
    // For simplicity, we assume batch_size = 1 in this implementation
    size_t seq_len = input.rows();
    
    // Project input to query, key, value
    Matrix<float> query = input * query_weights_; // [seq_len, model_dim]
    Matrix<float> key = input * key_weights_;     // [seq_len, model_dim]
    Matrix<float> value = input * value_weights_; // [seq_len, model_dim]
    
    // Reshape for multi-head attention
    // In a more sophisticated implementation, we would reshape to [batch_size, seq_len, num_heads, head_dim]
    // Here we'll compute attention for each head separately
    
    Matrix<float> output(seq_len, model_dim_);
    output.fill(0.0f);
    
    // Process each attention head
    for (int h = 0; h < num_heads_; ++h) {
        // Extract head-specific query, key, value
        Matrix<float> q_head(seq_len, head_dim_);
        Matrix<float> k_head(seq_len, head_dim_);
        Matrix<float> v_head(seq_len, head_dim_);
        
        // Extract the portion of Q, K, V for this head
        for (size_t i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim_; ++j) {
                q_head(i, j) = query(i, h * head_dim_ + j);
                k_head(i, j) = key(i, h * head_dim_ + j);
                v_head(i, j) = value(i, h * head_dim_ + j);
            }
        }
        
        // Compute attention for this head
        Matrix<float> head_output = scaled_dot_product_attention(q_head, k_head, v_head, mask);
        
        // Copy head output back to the combined output
        for (size_t i = 0; i < seq_len; ++i) {
            for (int j = 0; j < head_dim_; ++j) {
                output(i, h * head_dim_ + j) = head_output(i, j);
            }
        }
    }
    
    // Final projection
    return output * output_weights_;
}

Matrix<float> MultiHeadAttention::scaled_dot_product_attention(
    const Matrix<float>& Q, 
    const Matrix<float>& K, 
    const Matrix<float>& V,
    const Matrix<float>* mask) const {
    
    size_t seq_len = Q.rows();
    size_t d_k = Q.cols(); // head dimension
    
    // Compute attention scores: Q * K^T
    Matrix<float> scores = Q * K.transpose();
    
    // Scale by sqrt(d_k)
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));
    for (size_t i = 0; i < scores.size(); ++i) {
        scores.data()[i] *= scale_factor;
    }
    
    // Apply mask if provided (for causal attention)
    if (mask != nullptr) {
        if (mask->rows() != seq_len || mask->cols() != seq_len) {
            throw ModelException("Attention mask dimensions mismatch");
        }
        
        // Apply mask by adding large negative values to masked positions
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                if ((*mask)(i, j) == 0.0f) {
                    scores(i, j) = -1e9f; // Effectively -infinity for float
                }
            }
        }
    }
    
    // Apply softmax row-wise to get attention weights
    Matrix<float> attention_weights(seq_len, seq_len);
    
    for (size_t i = 0; i < seq_len; ++i) {
        // Find max value in the row for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < seq_len; ++j) {
            max_val = std::max(max_val, scores(i, j));
        }
        
        // Compute exp(x - max_val) for each element
        std::vector<float> exp_values(seq_len);
        float sum_exp = 0.0f;
        
        for (size_t j = 0; j < seq_len; ++j) {
            float exp_val = std::exp(scores(i, j) - max_val);
            exp_values[j] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize by sum to get softmax
        for (size_t j = 0; j < seq_len; ++j) {
            attention_weights(i, j) = exp_values[j] / sum_exp;
        }
    }
    
    // Compute weighted sum: attention_weights * V
    return attention_weights * V;
}

} // namespace tiny_llama