#include "tiny_llama/model.hpp"
#include "tiny_llama/exceptions.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

namespace tiny_llama {

// FeedForwardNetwork implementation
FeedForwardNetwork::FeedForwardNetwork(int model_dim, int hidden_dim) 
    : model_dim_(model_dim), hidden_dim_(hidden_dim) {
    // Initialize weight matrices
    linear1_weights_ = Matrix<float>(model_dim, hidden_dim);
    linear1_bias_.resize(hidden_dim, 0.0f);
    linear2_weights_ = Matrix<float>(hidden_dim, model_dim);
    linear2_bias_.resize(model_dim, 0.0f);
}

void FeedForwardNetwork::load_weights(const std::string& weights_file) {
    std::ifstream file(weights_file, std::ios::binary);
    if (!file.is_open()) {
        throw FileIOException("Cannot open weights file: " + weights_file);
    }
    
    try {
        // Read linear1 weights
        size_t rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
        
        if (rows != model_dim_ || cols != hidden_dim_) {
            throw FileIOException("Mismatch in linear1 weights dimensions");
        }
        
        linear1_weights_.resize(rows, cols);
        file.read(reinterpret_cast<char*>(linear1_weights_.data()), rows * cols * sizeof(float));
        
        // Read linear1 bias
        size_t bias1_size;
        file.read(reinterpret_cast<char*>(&bias1_size), sizeof(size_t));
        if (bias1_size != hidden_dim_) {
            throw FileIOException("Mismatch in linear1 bias dimensions");
        }
        linear1_bias_.resize(bias1_size);
        file.read(reinterpret_cast<char*>(linear1_bias_.data()), bias1_size * sizeof(float));
        
        // Read linear2 weights
        file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
        
        if (rows != hidden_dim_ || cols != model_dim_) {
            throw FileIOException("Mismatch in linear2 weights dimensions");
        }
        
        linear2_weights_.resize(rows, cols);
        file.read(reinterpret_cast<char*>(linear2_weights_.data()), rows * cols * sizeof(float));
        
        // Read linear2 bias
        size_t bias2_size;
        file.read(reinterpret_cast<char*>(&bias2_size), sizeof(size_t));
        if (bias2_size != model_dim_) {
            throw FileIOException("Mismatch in linear2 bias dimensions");
        }
        linear2_bias_.resize(bias2_size);
        file.read(reinterpret_cast<char*>(linear2_bias_.data()), bias2_size * sizeof(float));
        
    } catch (const std::exception& e) {
        throw FileIOException("Error loading FFN weights: " + std::string(e.what()));
    }
    
    file.close();
}

Matrix<float> FeedForwardNetwork::forward(const Matrix<float>& input) const {
    if (input.cols() != model_dim_) {
        throw ModelException("Input dimension mismatch for FFN: expected " + 
                            std::to_string(model_dim_) + ", got " + 
                            std::to_string(input.cols()));
    }
    
    // First linear layer: input * linear1_weights + bias
    Matrix<float> hidden = input * linear1_weights_;
    
    // Apply bias to each row
    for (size_t i = 0; i < hidden.rows(); ++i) {
        for (size_t j = 0; j < hidden.cols(); ++j) {
            hidden(i, j) += linear1_bias_[j];
        }
    }
    
    // Apply GELU activation
    for (size_t i = 0; i < hidden.rows(); ++i) {
        std::vector<float> row_data(hidden_dim_);
        for (size_t j = 0; j < hidden_dim_; ++j) {
            row_data[j] = hidden(i, j);
        }
        
        std::vector<float> activated = gelu_activation(row_data);
        
        for (size_t j = 0; j < hidden_dim_; ++j) {
            hidden(i, j) = activated[j];
        }
    }
    
    // Second linear layer: hidden * linear2_weights + bias
    Matrix<float> output = hidden * linear2_weights_;
    
    // Apply bias to each row
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            output(i, j) += linear2_bias_[j];
        }
    }
    
    return output;
}

std::vector<float> FeedForwardNetwork::gelu_activation(const std::vector<float>& input) const {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // This is the approximation used in the paper "Gaussian Error Linear Units (GELUs)"
    
    const float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/π)
    const float coeff =0.044715f;
    
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

// TransformerBlock implementation
TransformerBlock::TransformerBlock(int model_dim, int num_heads, int ffn_hidden_dim) 
    : model_dim_(model_dim) {
    attention_ = std::make_unique<MultiHeadAttention>(model_dim, num_heads);
    ffn_ = std::make_unique<FeedForwardNetwork>(model_dim, ffn_hidden_dim);
    
    // Initialize layer norm parameters
    layer_norm1_weight_.resize(model_dim, 1.0f);
    layer_norm1_bias_.resize(model_dim, 0.0f);
    layer_norm2_weight_.resize(model_dim, 1.0f);
    layer_norm2_bias_.resize(model_dim, 0.0f);
}

void TransformerBlock::load_weights(const std::string& weights_file) {
    try {
        // Load attention weights
        attention_->load_weights(weights_file + ".attention");
        
        // Load FFN weights
        ffn_->load_weights(weights_file + ".ffn");
        
        // Load layer norm weights
        std::ifstream file(weights_file + ".layernorm", std::ios::binary);
        if (!file.is_open()) {
            throw FileIOException("Cannot open layer norm weights file: " + weights_file + ".layernorm");
        }
        
        // Read layer norm 1 weights
        size_t ln1_size;
        file.read(reinterpret_cast<char*>(&ln1_size), sizeof(size_t));
        if (ln1_size != model_dim_) {
            throw FileIOException("Mismatch in layer norm 1 weights dimensions");
        }
        layer_norm1_weight_.resize(ln1_size);
        file.read(reinterpret_cast<char*>(layer_norm1_weight_.data()), ln1_size * sizeof(float));
        
        // Read layer norm 1 bias
        size_t ln1_bias_size;
        file.read(reinterpret_cast<char*>(&ln1_bias_size), sizeof(size_t));
        if (ln1_bias_size != model_dim_) {
            throw FileIOException("Mismatch in layer norm 1 bias dimensions");
        }
        layer_norm1_bias_.resize(ln1_bias_size);
        file.read(reinterpret_cast<char*>(layer_norm1_bias_.data()), ln1_bias_size * sizeof(float));
        
        // Read layer norm 2 weights
        size_t ln2_size;
        file.read(reinterpret_cast<char*>(&ln2_size), sizeof(size_t));
        if (ln2_size != model_dim_) {
            throw FileIOException("Mismatch in layer norm 2weights dimensions");
        }
        layer_norm2_weight_.resize(ln2_size);
        file.read(reinterpret_cast<char*>(layer_norm2_weight_.data()), ln2_size * sizeof(float));
        
        // Read layer norm 2 bias
        size_t ln2_bias_size;
        file.read(reinterpret_cast<char*>(&ln2_bias_size), sizeof(size_t));
        if (ln2_bias_size != model_dim_) {
            throw FileIOException("Mismatch in layer norm 2 bias dimensions");
        }
        layer_norm2_bias_.resize(ln2_bias_size);
        file.read(reinterpret_cast<char*>(layer_norm2_bias_.data()), ln2_bias_size * sizeof(float));
        
        file.close();
    } catch (const std::exception& e) {
        throw FileIOException("Error loading transformer block weights: " + std::string(e.what()));
    }
}

Matrix<float> TransformerBlock::forward(const Matrix<float>& input, const Matrix<float>* mask) const {
    if (input.cols() != model_dim_) {
        throw ModelException("Input dimension mismatch for transformer block: expected " + 
                            std::to_string(model_dim_) + ", got " + 
                            std::to_string(input.cols()));
    }
    
    // First sub-layer: Multi-head attention with residual connection and layer norm
    Matrix<float> normalized_input = layer_norm(input, layer_norm1_weight_, layer_norm1_bias_);
    Matrix<float> attention_output = attention_->forward(normalized_input, mask);
    
    // Residual connection
    Matrix<float> residual1 = input;
    for (size_t i = 0; i < residual1.rows(); ++i) {
        for (size_t j = 0; j < residual1.cols(); ++j) {
            residual1(i, j) += attention_output(i, j);
        }
    }
    
    // Second sub-layer: Feed-forward network with residual connection and layer norm
    Matrix<float> normalized_residual = layer_norm(residual1, layer_norm2_weight_, layer_norm2_bias_);
    Matrix<float> ffn_output = ffn_->forward(normalized_residual);
    
    // Residual connection
    Matrix<float> output = residual1;
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            output(i, j) += ffn_output(i, j);
        }
    }
    
    return output;
}

Matrix<float> TransformerBlock::layer_norm(const Matrix<float>& input, 
                                          const std::vector<float>& weight,
                                          const std::vector<float>& bias) const {
    if (input.cols() != model_dim_) {
        throw ModelException("Input dimension mismatch for layer normalization");
    }
    
    if (weight.size() != model_dim_ || bias.size() != model_dim_) {
        throw ModelException("Weight or bias dimension mismatch for layer normalization");
    }
    
    Matrix<float> output(input.rows(), input.cols());
    
    // Layer normalization is applied to each row (sequence position) independently
    for (size_t i = 0; i < input.rows(); ++i) {
        // Calculate mean
        float mean = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            mean += input(i, j);
        }
        mean /= input.cols();
        
        // Calculate variance
        float variance = 0.0f;
        for (size_t j = 0; j < input.cols(); ++j) {
            float diff = input(i, j) - mean;
            variance += diff * diff;
        }
        variance /= input.cols();
        
        // Small epsilon to avoid division by zero
        const float epsilon = 1e-5f;
        
        // Normalize, scale, and shift
        for (size_t j = 0; j < input.cols(); ++j) {
            output(i, j) = (input(i, j) - mean) / std::sqrt(variance + epsilon) * weight[j] + bias[j];
        }
    }
    
    return output;
}

} // namespace tiny_llama