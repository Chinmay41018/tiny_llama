#pragma once

#include <string>
#include <vector>
#include <memory>

namespace tiny_llama {

// Forward declarations
class TinyLlamaModel;

/**
 * @brief Main public interface for the Tiny Llama C++ module
 * 
 * This class provides a simple API for text tokenization and generation
 * using a lightweight transformer-based language model.
 */
class TinyLlama {
private:
    std::unique_ptr<TinyLlamaModel> model_;
    bool is_initialized_;
    
public:
    /**
     * @brief Default constructor
     */
    TinyLlama();
    
    /**
     * @brief Destructor
     */
    ~TinyLlama();
    
    // Non-copyable but movable
    TinyLlama(const TinyLlama&) = delete;
    TinyLlama& operator=(const TinyLlama&) = delete;
    TinyLlama(TinyLlama&&) = default;
    TinyLlama& operator=(TinyLlama&&) = default;
    
    /**
     * @brief Initialize the model from a model directory path
     * @param model_path Path to directory containing model files
     */
    void initialize(const std::string& model_path);
    
    /**
     * @brief Initialize with specific file paths
     * @param vocab_file Path to vocabulary file
     * @param merges_file Path to BPE merges file
     * @param weights_file Path to model weights file
     */
    void initialize_with_config(const std::string& vocab_file,
                               const std::string& merges_file,
                               const std::string& weights_file);
    
    /**
     * @brief Generate text from a prompt
     * @param prompt Input text prompt
     * @param max_tokens Maximum number of tokens to generate
     * @return Generated text
     */
    std::string generate(const std::string& prompt, int max_tokens = 50);
    
    /**
     * @brief Tokenize text to string tokens
     * @param text Input text
     * @return Vector of token strings
     */
    std::vector<std::string> tokenize_to_strings(const std::string& text) const;
    
    /**
     * @brief Tokenize text to token IDs
     * @param text Input text
     * @return Vector of token IDs
     */
    std::vector<int> tokenize_to_ids(const std::string& text) const;
    
    /**
     * @brief Convert token IDs back to text
     * @param token_ids Vector of token IDs
     * @return Decoded text
     */
    std::string detokenize(const std::vector<int>& token_ids) const;
    
    /**
     * @brief Set generation temperature
     * @param temperature Temperature value (higher = more random)
     */
    void set_temperature(float temperature);
    
    /**
     * @brief Set maximum sequence length
     * @param max_length Maximum sequence length
     */
    void set_max_sequence_length(int max_length);
    
    /**
     * @brief Check if model is ready for use
     * @return True if initialized and ready
     */
    bool is_ready() const { return is_initialized_; }
    
    /**
     * @brief Get vocabulary size
     * @return Size of the vocabulary
     */
    size_t get_vocab_size() const;
};

} // namespace tiny_llama