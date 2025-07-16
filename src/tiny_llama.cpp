#include "tiny_llama/tiny_llama.hpp"
#include "tiny_llama/model.hpp"
#include "tiny_llama/exceptions.hpp"
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include <limits>
#include <cmath>

namespace tiny_llama {

// Helper function to check if file exists (C++14 compatible)
bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// Helper function to check if path is a directory
bool is_directory(const std::string& path) {
    struct stat buffer;
    if (stat(path.c_str(), &buffer) != 0) {
        return false;
    }
    return S_ISDIR(buffer.st_mode);
}

// Helper function to validate string input
void validate_string_input(const std::string& input, const std::string& param_name, bool allow_empty = false) {
    if (!allow_empty && input.empty()) {
        TINY_LLAMA_THROW(ConfigurationException, param_name + " cannot be empty", param_name);
    }
    
    // Check for null characters which could cause issues
    if (input.find('\0') != std::string::npos) {
        TINY_LLAMA_THROW(ConfigurationException, param_name + " contains null characters", param_name);
    }
    
    // Check for extremely long strings that could cause memory issues
    const size_t MAX_STRING_LENGTH = 1000000; // 1MB limit
    if (input.length() > MAX_STRING_LENGTH) {
        TINY_LLAMA_THROW(ConfigurationException, 
                        param_name + " is too long (max " + std::to_string(MAX_STRING_LENGTH) + " characters)", 
                        param_name);
    }
}

// Helper function to validate file path
void validate_file_path(const std::string& path, const std::string& param_name) {
    if (path.empty()) {
        TINY_LLAMA_THROW(FileIOException, param_name + " cannot be empty", param_name);
    }
    
    // Check for null characters which could cause issues
    if (path.find('\0') != std::string::npos) {
        TINY_LLAMA_THROW(FileIOException, param_name + " contains null characters", param_name);
    }
    
    // Check for extremely long strings that could cause memory issues
    const size_t MAX_STRING_LENGTH = 1000000; // 1MB limit
    if (path.length() > MAX_STRING_LENGTH) {
        TINY_LLAMA_THROW(FileIOException, 
                        param_name + " is too long (max " + std::to_string(MAX_STRING_LENGTH) + " characters)", 
                        param_name);
    }
    
    if (!file_exists(path)) {
        TINY_LLAMA_THROW(FileIOException, "File does not exist: " + path, path);
    }
    
    // Check if it's actually a file and not a directory
    if (is_directory(path)) {
        TINY_LLAMA_THROW(FileIOException, "Path is a directory, not a file: " + path, path);
    }
}

// Helper function to validate directory path
void validate_directory_path(const std::string& path, const std::string& param_name) {
    if (path.empty()) {
        TINY_LLAMA_THROW(FileIOException, param_name + " cannot be empty", param_name);
    }
    
    // Check for null characters which could cause issues
    if (path.find('\0') != std::string::npos) {
        TINY_LLAMA_THROW(FileIOException, param_name + " contains null characters", param_name);
    }
    
    // Check for extremely long strings that could cause memory issues
    const size_t MAX_STRING_LENGTH = 1000000; // 1MB limit
    if (path.length() > MAX_STRING_LENGTH) {
        TINY_LLAMA_THROW(FileIOException, 
                        param_name + " is too long (max " + std::to_string(MAX_STRING_LENGTH) + " characters)", 
                        param_name);
    }
    
    if (!file_exists(path)) {
        TINY_LLAMA_THROW(FileIOException, "Directory does not exist: " + path, path);
    }
    
    if (!is_directory(path)) {
        TINY_LLAMA_THROW(FileIOException, "Path is not a directory: " + path, path);
    }
}

// Helper function to validate integer parameters
void validate_positive_integer(int value, const std::string& param_name, int min_value = 1) {
    if (value < min_value) {
        TINY_LLAMA_THROW(ConfigurationException, 
                        param_name + " must be at least " + std::to_string(min_value) + 
                        " (got " + std::to_string(value) + ")", 
                        param_name);
    }
    
    // Check for reasonable upper bounds to prevent memory issues
    const int MAX_REASONABLE_VALUE = 1000000;
    if (value > MAX_REASONABLE_VALUE) {
        TINY_LLAMA_THROW(ConfigurationException, 
                        param_name + " is too large (max " + std::to_string(MAX_REASONABLE_VALUE) + 
                        ", got " + std::to_string(value) + ")", 
                        param_name);
    }
}

// Helper function to validate float parameters
void validate_positive_float(float value, const std::string& param_name, float min_value = 0.0f) {
    if (value < min_value) {
        TINY_LLAMA_THROW(ConfigurationException, 
                        param_name + " must be at least " + std::to_string(min_value) + 
                        " (got " + std::to_string(value) + ")", 
                        param_name);
    }
    
    if (!std::isfinite(value)) {
        TINY_LLAMA_THROW(ConfigurationException, 
                        param_name + " must be a finite number (got " + std::to_string(value) + ")", 
                        param_name);
    }
    
    // Check for reasonable upper bounds
    const float MAX_REASONABLE_VALUE = 1000.0f;
    if (value > MAX_REASONABLE_VALUE) {
        TINY_LLAMA_THROW(ConfigurationException, 
                        param_name + " is too large (max " + std::to_string(MAX_REASONABLE_VALUE) + 
                        ", got " + std::to_string(value) + ")", 
                        param_name);
    }
}

// Helper function to validate token IDs vector
void validate_token_ids(const std::vector<int>& token_ids, const std::string& param_name) {
    if (token_ids.empty()) {
        return; // Empty vector is allowed for some operations
    }
    
    // Check for reasonable size limits
    const size_t MAX_TOKEN_COUNT = 100000;
    if (token_ids.size() > MAX_TOKEN_COUNT) {
        TINY_LLAMA_THROW(ConfigurationException, 
                        param_name + " contains too many tokens (max " + std::to_string(MAX_TOKEN_COUNT) + 
                        ", got " + std::to_string(token_ids.size()) + ")", 
                        param_name);
    }
    
    // Check for negative token IDs which are invalid
    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (token_ids[i] < 0) {
            TINY_LLAMA_THROW(ConfigurationException, 
                            param_name + " contains negative token ID at index " + std::to_string(i) + 
                            " (value: " + std::to_string(token_ids[i]) + ")", 
                            param_name);
        }
    }
}

TinyLlama::TinyLlama() : is_initialized_(false) {
    model_ = std::make_unique<TinyLlamaModel>();
}

TinyLlama::~TinyLlama() = default;

void TinyLlama::initialize(const std::string& model_path) {
    // Validate input parameters
    validate_directory_path(model_path, "model_path");
    
    // Check if already initialized
    if (is_initialized_) {
        TINY_LLAMA_THROW(ModelException, "Model is already initialized. Create a new instance to reinitialize.", "initialization_state");
    }
    
    // Construct expected file paths
    std::string vocab_file = model_path + "/vocab.txt";
    std::string merges_file = model_path + "/merges.txt";
    std::string weights_file = model_path + "/weights.bin";
    
    // Validate all required files exist
    validate_file_path(vocab_file, "vocab_file");
    validate_file_path(merges_file, "merges_file");
    validate_file_path(weights_file, "weights_file");
    
    initialize_with_config(vocab_file, merges_file, weights_file);
}

void TinyLlama::initialize_with_config(const std::string& vocab_file,
                                      const std::string& merges_file,
                                      const std::string& weights_file) {
    // Validate input parameters
    validate_file_path(vocab_file, "vocab_file");
    validate_file_path(merges_file, "merges_file");
    validate_file_path(weights_file, "weights_file");
    
    // Check if already initialized
    if (is_initialized_) {
        TINY_LLAMA_THROW(ModelException, "Model is already initialized. Create a new instance to reinitialize.", "initialization_state");
    }
    
    if (!model_) {
        TINY_LLAMA_THROW(ModelException, "Model not properly constructed", "model_state");
    }
    
    try {
        // Load tokenizer first
        model_->load_tokenizer(vocab_file, merges_file);
        
        // Load model weights
        model_->load_model_weights(weights_file);
        
        is_initialized_ = true;
    } catch (const TinyLlamaException& e) {
        is_initialized_ = false;
        throw; // Re-throw TinyLlama exceptions as-is
    } catch (const std::exception& e) {
        is_initialized_ = false;
        TINY_LLAMA_THROW(ModelException, "Failed to initialize model: " + std::string(e.what()), "initialization");
    }
}

std::string TinyLlama::generate(const std::string& prompt, int max_tokens) {
    // Check initialization status
    if (!is_initialized_) {
        TINY_LLAMA_THROW(ModelException, "Model not initialized. Call initialize() first.", "initialization_state");
    }
    
    // Validate input parameters
    validate_string_input(prompt, "prompt", false); // Don't allow empty prompts
    validate_positive_integer(max_tokens, "max_tokens", 1);
    
    // Additional validation for max_tokens upper bound
    const int MAX_GENERATION_TOKENS = 10000;
    if (max_tokens > MAX_GENERATION_TOKENS) {
        TINY_LLAMA_THROW(ConfigurationException, 
                        "max_tokens is too large (max " + std::to_string(MAX_GENERATION_TOKENS) + 
                        ", got " + std::to_string(max_tokens) + ")", 
                        "max_tokens");
    }
    
    // Check if max_tokens exceeds the model's configured maximum sequence length
    int model_max_tokens = model_->get_config().max_sequence_length;
    if (max_tokens > model_max_tokens) {
        TINY_LLAMA_THROW(ConfigurationException, 
                        "max_tokens exceeds model's configured maximum sequence length (model max: " + 
                        std::to_string(model_max_tokens) + ", requested: " + std::to_string(max_tokens) + ")", 
                        "max_tokens");
    }
    
    try {
        return model_->generate_text(prompt, max_tokens);
    } catch (const TinyLlamaException& e) {
        throw; // Re-throw TinyLlama exceptions as-is
    } catch (const std::exception& e) {
        TINY_LLAMA_THROW(ModelException, "Text generation failed: " + std::string(e.what()), "generation");
    }
}

std::vector<std::string> TinyLlama::tokenize_to_strings(const std::string& text) const {
    // Check initialization status
    if (!is_initialized_) {
        TINY_LLAMA_THROW(TokenizerException, "Model not initialized. Call initialize() first.", "initialization_state");
    }
    
    // Validate input parameters (allow empty text for tokenization)
    validate_string_input(text, "text", true);
    
    try {
        return model_->tokenize_to_strings(text);
    } catch (const TinyLlamaException& e) {
        throw; // Re-throw TinyLlama exceptions as-is
    } catch (const std::exception& e) {
        TINY_LLAMA_THROW(TokenizerException, "Tokenization failed: " + std::string(e.what()), "tokenization");
    }
}

std::vector<int> TinyLlama::tokenize_to_ids(const std::string& text) const {
    // Check initialization status
    if (!is_initialized_) {
        TINY_LLAMA_THROW(TokenizerException, "Model not initialized. Call initialize() first.", "initialization_state");
    }
    
    // Validate input parameters (allow empty text for tokenization)
    validate_string_input(text, "text", true);
    
    try {
        return model_->tokenize(text);
    } catch (const TinyLlamaException& e) {
        throw; // Re-throw TinyLlama exceptions as-is
    } catch (const std::exception& e) {
        TINY_LLAMA_THROW(TokenizerException, "Tokenization failed: " + std::string(e.what()), "tokenization");
    }
}

std::string TinyLlama::detokenize(const std::vector<int>& token_ids) const {
    // Check initialization status
    if (!is_initialized_) {
        TINY_LLAMA_THROW(TokenizerException, "Model not initialized. Call initialize() first.", "initialization_state");
    }
    
    // Validate input parameters
    validate_token_ids(token_ids, "token_ids");
    
    try {
        return model_->detokenize(token_ids);
    } catch (const TinyLlamaException& e) {
        throw; // Re-throw TinyLlama exceptions as-is
    } catch (const std::exception& e) {
        TINY_LLAMA_THROW(TokenizerException, "Detokenization failed: " + std::string(e.what()), "detokenization");
    }
}

void TinyLlama::set_temperature(float temperature) {
    if (!model_) {
        TINY_LLAMA_THROW(ConfigurationException, "Model not properly constructed", "model_state");
    }
    
    // Validate temperature parameter (must be positive, reasonable upper bound)
    validate_positive_float(temperature, "temperature", 0.01f); // Minimum 0.01 to avoid division issues
    
    try {
        model_->set_temperature(temperature);
    } catch (const TinyLlamaException& e) {
        throw; // Re-throw TinyLlama exceptions as-is
    } catch (const std::exception& e) {
        TINY_LLAMA_THROW(ConfigurationException, "Failed to set temperature: " + std::string(e.what()), "temperature");
    }
}

void TinyLlama::set_max_sequence_length(int max_length) {
    if (!model_) {
        TINY_LLAMA_THROW(ConfigurationException, "Model not properly constructed", "model_state");
    }
    
    // Validate max_length parameter
    validate_positive_integer(max_length, "max_length", 1);
    
    // Additional validation for reasonable sequence length bounds
    const int MAX_SEQUENCE_LENGTH = 100000;
    if (max_length > MAX_SEQUENCE_LENGTH) {
        TINY_LLAMA_THROW(ConfigurationException, 
                        "max_length is too large (max " + std::to_string(MAX_SEQUENCE_LENGTH) + 
                        ", got " + std::to_string(max_length) + ")", 
                        "max_length");
    }
    
    // Note: This would require modifying the model configuration
    // For now, we'll throw an exception indicating this needs to be set during initialization
    TINY_LLAMA_THROW(ConfigurationException, 
                    "Max sequence length must be set during model initialization. Current implementation does not support runtime changes.", 
                    "runtime_configuration");
}

size_t TinyLlama::get_vocab_size() const {
    if (!model_) {
        TINY_LLAMA_THROW(ModelException, "Model not properly constructed", "model_state");
    }
    
    if (!is_initialized_) {
        TINY_LLAMA_THROW(ModelException, "Model not initialized. Call initialize() first.", "initialization_state");
    }
    
    try {
        return model_->get_vocab_size();
    } catch (const TinyLlamaException& e) {
        throw; // Re-throw TinyLlama exceptions as-is
    } catch (const std::exception& e) {
        TINY_LLAMA_THROW(ModelException, "Failed to get vocabulary size: " + std::string(e.what()), "vocab_size");
    }
}

} // namespace tiny_llama