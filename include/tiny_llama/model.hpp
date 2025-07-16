#pragma once

#include "matrix.hpp"
#include "tokenizer.hpp"
#include <memory>
#include <vector>
#include <fstream>

namespace tiny_llama {

/**
 * @brief Multi-head attention mechanism
 */
class MultiHeadAttention {
    friend class TinyLlamaModel;  // Allow TinyLlamaModel to access private members
    friend class WeightTestHelper;  // Allow test helper to access private members
    friend class TestAttention;  // Allow test class to access private members
private:
    Matrix<float> query_weights_;
    Matrix<float> key_weights_;
    Matrix<float> value_weights_;
    Matrix<float> output_weights_;
    int num_heads_;
    int head_dim_;
    int model_dim_;
    
protected:
    /**
     * @brief Compute scaled dot-product attention
     * @param Q Query matrix
     * @param K Key matrix
     * @param V Value matrix
     * @param mask Optional attention mask
     * @return Attention output
     */
    Matrix<float> scaled_dot_product_attention(
        const Matrix<float>& Q, 
        const Matrix<float>& K, 
        const Matrix<float>& V,
        const Matrix<float>* mask = nullptr) const;
    
public:
    /**
     * @brief Constructor
     * @param model_dim Model dimension
     * @param num_heads Number of attention heads
     */
    MultiHeadAttention(int model_dim, int num_heads);
    
    /**
     * @brief Load weights from file
     * @param weights_file Path to weights file
     */
    void load_weights(const std::string& weights_file);
    
    /**
     * @brief Forward pass
     * @param input Input matrix
     * @param mask Optional attention mask
     * @return Output matrix
     */
    Matrix<float> forward(const Matrix<float>& input, 
                         const Matrix<float>* mask = nullptr) const;
    
    /**
     * @brief Get model dimension
     * @return Model dimension
     */
    int get_model_dim() const { return model_dim_; }
    
    /**
     * @brief Get number of heads
     * @return Number of attention heads
     */
    int get_num_heads() const { return num_heads_; }
};

/**
 * @brief Feed-forward network
 */
class FeedForwardNetwork {
    friend class TinyLlamaModel;  // Allow TinyLlamaModel to access private members
    friend class WeightTestHelper;  // Allow test helper to access private members
private:
    Matrix<float> linear1_weights_;
    std::vector<float> linear1_bias_;
    Matrix<float> linear2_weights_;
    std::vector<float> linear2_bias_;
    int model_dim_;
    int hidden_dim_;
    
    /**
     * @brief GELU activation function
     * @param input Input vector
     * @return Activated vector
     */
    std::vector<float> gelu_activation(const std::vector<float>& input) const;
    
public:
    /**
     * @brief Constructor
     * @param model_dim Model dimension
     * @param hidden_dim Hidden layer dimension
     */
    FeedForwardNetwork(int model_dim, int hidden_dim);
    
    /**
     * @brief Load weights from file
     * @param weights_file Path to weights file
     */
    void load_weights(const std::string& weights_file);
    
    /**
     * @brief Forward pass
     * @param input Input matrix
     * @return Output matrix
     */
    Matrix<float> forward(const Matrix<float>& input) const;
    
    /**
     * @brief Get model dimension
     * @return Model dimension
     */
    int get_model_dim() const { return model_dim_; }
    
    /**
     * @brief Get hidden dimension
     * @return Hidden dimension
     */
    int get_hidden_dim() const { return hidden_dim_; }
};

/**
 * @brief Transformer block combining attention and feed-forward layers
 */
class TransformerBlock {
    friend class TinyLlamaModel;  // Allow TinyLlamaModel to access private members
    friend class WeightTestHelper;  // Allow test helper to access private members
private:
    std::unique_ptr<MultiHeadAttention> attention_;
    std::unique_ptr<FeedForwardNetwork> ffn_;
    std::vector<float> layer_norm1_weight_;
    std::vector<float> layer_norm1_bias_;
    std::vector<float> layer_norm2_weight_;
    std::vector<float> layer_norm2_bias_;
    int model_dim_;
    
    /**
     * @brief Layer normalization
     * @param input Input matrix
     * @param weight Normalization weights
     * @param bias Normalization bias
     * @return Normalized matrix
     */
    Matrix<float> layer_norm(const Matrix<float>& input, 
                            const std::vector<float>& weight,
                            const std::vector<float>& bias) const;
    
public:
    /**
     * @brief Constructor
     * @param model_dim Model dimension
     * @param num_heads Number of attention heads
     * @param ffn_hidden_dim Feed-forward hidden dimension
     */
    TransformerBlock(int model_dim, int num_heads, int ffn_hidden_dim);
    
    /**
     * @brief Load weights from file
     * @param weights_file Path to weights file
     */
    void load_weights(const std::string& weights_file);
    
    /**
     * @brief Forward pass
     * @param input Input matrix
     * @param mask Optional attention mask
     * @return Output matrix
     */
    Matrix<float> forward(const Matrix<float>& input, 
                         const Matrix<float>* mask = nullptr) const;
    
    /**
     * @brief Get model dimension
     * @return Model dimension
     */
    int get_model_dim() const { return model_dim_; }
};

/**
 * @brief Model configuration structure
 */
struct ModelConfig {
    int model_dim = 512;
    int num_layers = 6;
    int num_heads = 8;
    int ffn_hidden_dim = 2048;
    int max_sequence_length = 1024;
    int vocab_size = 32000;
    float dropout_rate = 0.1f;
    
    /**
     * @brief Load configuration from file
     * @param config_file Path to configuration file
     */
    void load_from_file(const std::string& config_file);
    
    /**
     * @brief Save configuration to file
     * @param config_file Path to configuration file
     */
    void save_to_file(const std::string& config_file) const;
};

/**
 * @brief Main Tiny Llama model implementation
 */
class TinyLlamaModel {
    friend class WeightTestHelper;  // Allow test helper to access private members
private:
    std::unique_ptr<BPETokenizer> tokenizer_;
    Matrix<float> embedding_weights_;
    Matrix<float> position_embeddings_;
    std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks_;
    Matrix<float> output_projection_;
    
    ModelConfig config_;
    float temperature_;
    
    /**
     * @brief Create causal attention mask
     * @param sequence_length Length of sequence
     * @return Attention mask matrix
     */
    Matrix<float> create_attention_mask(int sequence_length) const;
    
    /**
     * @brief Apply softmax to logits
     * @param logits Input logits
     * @param temperature Sampling temperature (default uses class member)
     * @return Probability distribution
     */
    std::vector<float> softmax(const std::vector<float>& logits, float temperature = -1.0f) const;
    
    /**
     * @brief Sample token from probability distribution using greedy sampling (argmax)
     * @param probabilities Token probabilities
     * @return Sampled token ID (highest probability token)
     */
    int sample_token(const std::vector<float>& probabilities) const;
    
    /**
     * @brief Load attention weights for a specific layer from binary stream
     * @param file Input file stream
     * @param layer_index Layer index
     */
    void load_attention_weights(std::ifstream& file, int layer_index);
    
    /**
     * @brief Load FFN weights for a specific layer from binary stream
     * @param file Input file stream
     * @param layer_index Layer index
     */
    void load_ffn_weights(std::ifstream& file, int layer_index);
    
    /**
     * @brief Load layer normalization weights for a specific layer from binary stream
     * @param file Input file stream
     * @param layer_index Layer index
     */
    void load_layer_norm_weights(std::ifstream& file, int layer_index);
    

    
public:
    /**
     * @brief Constructor with default configuration
     */
    TinyLlamaModel();
    
    /**
     * @brief Constructor with custom configuration
     * @param config Model configuration
     */
    explicit TinyLlamaModel(const ModelConfig& config);
    
    /**
     * @brief Load tokenizer from files
     * @param vocab_file Path to vocabulary file
     * @param merges_file Path to BPE merges file
     */
    void load_tokenizer(const std::string& vocab_file, const std::string& merges_file);
    
    /**
     * @brief Load model weights from file
     * @param weights_file Path to weights file
     */
    void load_model_weights(const std::string& weights_file);
    
    /**
     * @brief Forward pass through model
     * @param input_tokens Input token IDs
     * @return Output logits
     */
    std::vector<float> forward(const std::vector<int>& input_tokens) const;
    
    /**
     * @brief Generate text from prompt
     * @param prompt Input text prompt
     * @param max_tokens Maximum tokens to generate
     * @param temperature Sampling temperature
     * @return Generated text
     */
    std::string generate_text(const std::string& prompt, int max_tokens = 50, 
                             float temperature = 1.0f) const;
    
    /**
     * @brief Tokenize text
     * @param text Input text
     * @return Token IDs
     */
    std::vector<int> tokenize(const std::string& text) const;
    
    /**
     * @brief Detokenize token IDs
     * @param tokens Token IDs
     * @return Text string
     */
    std::string detokenize(const std::vector<int>& tokens) const;
    
    /**
     * @brief Tokenize to string tokens
     * @param text Input text
     * @return Token strings
     */
    std::vector<std::string> tokenize_to_strings(const std::string& text) const;
    
    /**
     * @brief Set generation temperature
     * @param temperature New temperature value
     */
    void set_temperature(float temperature) { temperature_ = temperature; }
    
    /**
     * @brief Get current temperature
     * @return Current temperature
     */
    float get_temperature() const { return temperature_; }
    
    /**
     * @brief Get model configuration
     * @return Model configuration
     */
    const ModelConfig& get_config() const { return config_; }
    
    /**
     * @brief Get vocabulary size
     * @return Vocabulary size
     */
    size_t get_vocab_size() const;
    
    /**
     * @brief Check if model is initialized
     * @return True if model is ready
     */
    bool is_initialized() const;
    
    /**
     * @brief Save model weights to binary file (for testing purposes)
     * @param weights_file Path to output weights file
     */
    void save_model_weights(const std::string& weights_file) const;
};

} // namespace tiny_llama