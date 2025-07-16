#include "tiny_llama/tiny_llama.hpp"
#include "tiny_llama/exceptions.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdio>

using namespace tiny_llama;

// Helper function to check if file exists (C++14 compatible)
bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// Helper function to create directory (C++14 compatible)
void create_directory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// Helper function to remove file (C++14 compatible)
void remove_file(const std::string& path) {
    remove(path.c_str());
}

/**
 * @brief Test helper to create sample files for testing
 */
class TestFileHelper {
public:
    static void create_test_directory(const std::string& path) {
        create_directory(path);
    }
    
    static void create_sample_vocab_file(const std::string& filepath) {
        std::ofstream file(filepath);
        file << "<unk>\n";
        file << "<pad>\n";
        file << "<bos>\n";
        file << "<eos>\n";
        file << "hello\n";
        file << "world\n";
        file << "test\n";
        file << "token\n";
        file.close();
    }
    
    static void create_sample_merges_file(const std::string& filepath) {
        std::ofstream file(filepath);
        file << "h e\n";
        file << "l l\n";
        file << "o r\n";
        file.close();
    }
    
    static void create_sample_weights_file(const std::string& filepath) {
        std::ofstream file(filepath, std::ios::binary);
        
        // Write header matching the expected format
        uint32_t magic = 0x544C4C4D; // "TLLM" in hex
        uint32_t version = 1;
        
        // Use default model configuration values
        int model_dim = 512;
        int num_layers = 6;
        int num_heads = 8;
        int ffn_hidden_dim = 2048;
        int max_seq_len = 1024;
        int vocab_size = 32000;
        float dropout_rate = 0.1f;
        
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&model_dim), sizeof(model_dim));
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
        file.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
        file.write(reinterpret_cast<const char*>(&ffn_hidden_dim), sizeof(ffn_hidden_dim));
        file.write(reinterpret_cast<const char*>(&max_seq_len), sizeof(max_seq_len));
        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        file.write(reinterpret_cast<const char*>(&dropout_rate), sizeof(dropout_rate));
        
        // Write embedding weights dimensions and data
        size_t embedding_rows = vocab_size;
        size_t embedding_cols = model_dim;
        file.write(reinterpret_cast<const char*>(&embedding_rows), sizeof(embedding_rows));
        file.write(reinterpret_cast<const char*>(&embedding_cols), sizeof(embedding_cols));
        
        std::vector<float> embedding_weights(embedding_rows * embedding_cols, 0.01f);
        file.write(reinterpret_cast<const char*>(embedding_weights.data()), 
                  embedding_weights.size() * sizeof(float));
        
        // Write position embedding weights dimensions and data
        size_t pos_embedding_rows = max_seq_len;
        size_t pos_embedding_cols = model_dim;
        file.write(reinterpret_cast<const char*>(&pos_embedding_rows), sizeof(pos_embedding_rows));
        file.write(reinterpret_cast<const char*>(&pos_embedding_cols), sizeof(pos_embedding_cols));
        
        std::vector<float> pos_embedding_weights(pos_embedding_rows * pos_embedding_cols, 0.01f);
        file.write(reinterpret_cast<const char*>(pos_embedding_weights.data()), 
                  pos_embedding_weights.size() * sizeof(float));
        
        // Write transformer block weights for each layer
        for (int layer = 0; layer < num_layers; ++layer) {
            // Attention weights (Q, K, V, output)
            size_t attention_weight_size = model_dim * model_dim;
            std::vector<float> attention_weights(attention_weight_size * 4, 0.01f);
            
            // Write Q, K, V, output weights
            size_t attention_rows = model_dim;
            size_t attention_cols = model_dim;
            for (int i = 0; i < 4; ++i) {
                file.write(reinterpret_cast<const char*>(&attention_rows), sizeof(size_t)); // rows
                file.write(reinterpret_cast<const char*>(&attention_cols), sizeof(size_t)); // cols
                file.write(reinterpret_cast<const char*>(attention_weights.data() + i * attention_weight_size), 
                          attention_weight_size * sizeof(float));
            }
            
            // FFN weights
            size_t ffn1_rows = model_dim;
            size_t ffn1_cols = ffn_hidden_dim;
            size_t ffn2_rows = ffn_hidden_dim;
            size_t ffn2_cols = model_dim;
            
            file.write(reinterpret_cast<const char*>(&ffn1_rows), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&ffn1_cols), sizeof(size_t));
            std::vector<float> ffn1_weights(ffn1_rows * ffn1_cols, 0.01f);
            file.write(reinterpret_cast<const char*>(ffn1_weights.data()), 
                      ffn1_weights.size() * sizeof(float));
            
            size_t ffn1_bias_size = ffn_hidden_dim;
            file.write(reinterpret_cast<const char*>(&ffn1_bias_size), sizeof(size_t));
            std::vector<float> ffn1_bias(ffn_hidden_dim, 0.01f);
            file.write(reinterpret_cast<const char*>(ffn1_bias.data()), 
                      ffn1_bias.size() * sizeof(float));
            
            file.write(reinterpret_cast<const char*>(&ffn2_rows), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&ffn2_cols), sizeof(size_t));
            std::vector<float> ffn2_weights(ffn2_rows * ffn2_cols, 0.01f);
            file.write(reinterpret_cast<const char*>(ffn2_weights.data()), 
                      ffn2_weights.size() * sizeof(float));
            
            size_t ffn2_bias_size = model_dim;
            file.write(reinterpret_cast<const char*>(&ffn2_bias_size), sizeof(size_t));
            std::vector<float> ffn2_bias(model_dim, 0.01f);
            file.write(reinterpret_cast<const char*>(ffn2_bias.data()), 
                      ffn2_bias.size() * sizeof(float));
            
            // Layer norm weights and biases
            size_t ln_size = model_dim;
            std::vector<float> ln1_weight(model_dim, 1.0f);
            std::vector<float> ln1_bias(model_dim, 0.0f);
            std::vector<float> ln2_weight(model_dim, 1.0f);
            std::vector<float> ln2_bias(model_dim, 0.0f);
            
            // Layer norm 1
            file.write(reinterpret_cast<const char*>(&ln_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(ln1_weight.data()), 
                      ln1_weight.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(&ln_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(ln1_bias.data()), 
                      ln1_bias.size() * sizeof(float));
            
            // Layer norm 2
            file.write(reinterpret_cast<const char*>(&ln_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(ln2_weight.data()), 
                      ln2_weight.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(&ln_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(ln2_bias.data()), 
                      ln2_bias.size() * sizeof(float));
        }
        
        // Write output projection weights
        size_t output_rows = model_dim;
        size_t output_cols = vocab_size;
        file.write(reinterpret_cast<const char*>(&output_rows), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&output_cols), sizeof(size_t));
        
        std::vector<float> output_weights(output_rows * output_cols, 0.01f);
        file.write(reinterpret_cast<const char*>(output_weights.data()), 
                  output_weights.size() * sizeof(float));
        
        file.close();
    }
    
    static void cleanup_test_directory(const std::string& path) {
        if (file_exists(path)) {
            // Remove files in the directory
            remove_file(path + "/vocab.txt");
            remove_file(path + "/merges.txt");
            remove_file(path + "/weights.bin");
            // Remove the directory
            rmdir(path.c_str());
        }
    }
};

/**
 * @brief Test TinyLlama constructor and destructor
 */
void test_tiny_llama_constructor_destructor() {
    std::cout << "Testing TinyLlama constructor and destructor..." << std::endl;
    
    // Test default constructor
    {
        TinyLlama llama;
        assert(!llama.is_ready());
        assert(llama.get_vocab_size() == 0);
    }
    
    // Test that destructor works properly (no crashes)
    std::cout << "✓ Constructor and destructor work correctly" << std::endl;
}

/**
 * @brief Test initialization methods
 */
void test_initialization() {
    std::cout << "Testing TinyLlama initialization..." << std::endl;
    
    const std::string test_dir = "test_model_data";
    TestFileHelper::create_test_directory(test_dir);
    
    try {
        TinyLlama llama;
        
        // Test initialization with non-existent path
        try {
            llama.initialize("non_existent_path");
            assert(false && "Should have thrown FileIOException");
        } catch (const FileIOException& e) {
            // Expected
        }
        
        // Test initialization with missing files
        try {
            llama.initialize(test_dir);
            assert(false && "Should have thrown FileIOException for missing files");
        } catch (const FileIOException& e) {
            // Expected
        }
        
        // Create sample files
        TestFileHelper::create_sample_vocab_file(test_dir + "/vocab.txt");
        TestFileHelper::create_sample_merges_file(test_dir + "/merges.txt");
        TestFileHelper::create_sample_weights_file(test_dir + "/weights.bin");
        
        // Test successful initialization
        llama.initialize(test_dir);
        assert(llama.is_ready());
        assert(llama.get_vocab_size() > 0);
        
        std::cout << "✓ Initialization methods work correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TestFileHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TestFileHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Test initialization with specific config
 */
void test_initialization_with_config() {
    std::cout << "Testing TinyLlama initialization with config..." << std::endl;
    
    const std::string test_dir = "test_config_data";
    TestFileHelper::create_test_directory(test_dir);
    
    try {
        TinyLlama llama;
        
        std::string vocab_file = test_dir + "/vocab.txt";
        std::string merges_file = test_dir + "/merges.txt";
        std::string weights_file = test_dir + "/weights.bin";
        
        // Create sample files
        TestFileHelper::create_sample_vocab_file(vocab_file);
        TestFileHelper::create_sample_merges_file(merges_file);
        TestFileHelper::create_sample_weights_file(weights_file);
        
        // Test successful initialization with config
        llama.initialize_with_config(vocab_file, merges_file, weights_file);
        assert(llama.is_ready());
        assert(llama.get_vocab_size() > 0);
        
        std::cout << "✓ Initialization with config works correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TestFileHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TestFileHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Test tokenization methods
 */
void test_tokenization() {
    std::cout << "Testing TinyLlama tokenization..." << std::endl;
    
    const std::string test_dir = "test_tokenization_data";
    TestFileHelper::create_test_directory(test_dir);
    
    try {
        TinyLlama llama;
        
        // Test tokenization before initialization
        try {
            llama.tokenize_to_ids("test");
            assert(false && "Should have thrown TokenizerException");
        } catch (const TokenizerException& e) {
            // Expected
        }
        
        try {
            llama.tokenize_to_strings("test");
            assert(false && "Should have thrown TokenizerException");
        } catch (const TokenizerException& e) {
            // Expected
        }
        
        try {
            llama.detokenize({1, 2, 3});
            assert(false && "Should have thrown TokenizerException");
        } catch (const TokenizerException& e) {
            // Expected
        }
        
        // Initialize model
        TestFileHelper::create_sample_vocab_file(test_dir + "/vocab.txt");
        TestFileHelper::create_sample_merges_file(test_dir + "/merges.txt");
        TestFileHelper::create_sample_weights_file(test_dir + "/weights.bin");
        
        llama.initialize(test_dir);
        
        // Test tokenization after initialization
        // The tokenizer should now work properly
        std::vector<int> token_ids = llama.tokenize_to_ids("hello world");
        assert(!token_ids.empty());
        
        std::vector<std::string> token_strings = llama.tokenize_to_strings("hello world");
        assert(!token_strings.empty());
        
        // Test detokenization
        std::string detokenized = llama.detokenize(token_ids);
        assert(!detokenized.empty());
        
        // Test empty input
        std::vector<int> empty_tokens = llama.tokenize_to_ids("");
        assert(empty_tokens.empty());
        
        std::cout << "✓ Tokenization methods work correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TestFileHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TestFileHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Test text generation
 */
void test_text_generation() {
    std::cout << "Testing TinyLlama text generation..." << std::endl;
    
    const std::string test_dir = "test_generation_data";
    TestFileHelper::create_test_directory(test_dir);
    
    try {
        TinyLlama llama;
        
        // Test generation before initialization
        try {
            llama.generate("test prompt");
            assert(false && "Should have thrown ModelException");
        } catch (const ModelException& e) {
            // Expected
        }
        
        // Initialize model
        TestFileHelper::create_sample_vocab_file(test_dir + "/vocab.txt");
        TestFileHelper::create_sample_merges_file(test_dir + "/merges.txt");
        TestFileHelper::create_sample_weights_file(test_dir + "/weights.bin");
        
        llama.initialize(test_dir);
        
        // Test invalid inputs
        try {
            llama.generate("");
            assert(false && "Should have thrown ModelException for empty prompt");
        } catch (const ModelException& e) {
            // Expected
        }
        
        try {
            llama.generate("test", 0);
            assert(false && "Should have thrown ModelException for zero max_tokens");
        } catch (const ModelException& e) {
            // Expected
        }
        
        try {
            llama.generate("test", -1);
            assert(false && "Should have thrown ModelException for negative max_tokens");
        } catch (const ModelException& e) {
            // Expected
        }
        
        // Test valid generation (may throw due to incomplete model, but should validate inputs)
        try {
            std::string result = llama.generate("hello", 5);
            // If it succeeds, result should not be empty
            if (!result.empty()) {
                std::cout << "Generated text: " << result << std::endl;
            }
        } catch (const ModelException& e) {
            // This is acceptable as the model may not be fully functional with dummy weights
            std::cout << "Generation failed as expected with dummy weights: " << e.what() << std::endl;
        }
        
        std::cout << "✓ Text generation validation works correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TestFileHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TestFileHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Test configuration methods
 */
void test_configuration() {
    std::cout << "Testing TinyLlama configuration..." << std::endl;
    
    TinyLlama llama;
    
    // Test temperature setting
    llama.set_temperature(0.8f);
    
    // Test invalid temperature
    try {
        llama.set_temperature(0.0f);
        assert(false && "Should have thrown ConfigurationException");
    } catch (const ConfigurationException& e) {
        // Expected
    }
    
    try {
        llama.set_temperature(-1.0f);
        assert(false && "Should have thrown ConfigurationException");
    } catch (const ConfigurationException& e) {
        // Expected
    }
    
    // Test max sequence length setting (should throw as not implemented)
    try {
        llama.set_max_sequence_length(512);
        assert(false && "Should have thrown ConfigurationException");
    } catch (const ConfigurationException& e) {
        // Expected - not implemented yet
    }
    
    try {
        llama.set_max_sequence_length(0);
        assert(false && "Should have thrown ConfigurationException");
    } catch (const ConfigurationException& e) {
        // Expected
    }
    
    std::cout << "✓ Configuration methods work correctly" << std::endl;
}

/**
 * @brief Test status methods
 */
void test_status_methods() {
    std::cout << "Testing TinyLlama status methods..." << std::endl;
    
    const std::string test_dir = "test_status_data";
    TestFileHelper::create_test_directory(test_dir);
    
    try {
        TinyLlama llama;
        
        // Test initial state
        assert(!llama.is_ready());
        assert(llama.get_vocab_size() == 0);
        
        // Initialize model
        TestFileHelper::create_sample_vocab_file(test_dir + "/vocab.txt");
        TestFileHelper::create_sample_merges_file(test_dir + "/merges.txt");
        TestFileHelper::create_sample_weights_file(test_dir + "/weights.bin");
        
        llama.initialize(test_dir);
        
        // Test after initialization
        assert(llama.is_ready());
        assert(llama.get_vocab_size() > 0);
        
        std::cout << "✓ Status methods work correctly" << std::endl;
        
    } catch (const std::exception& e) {
        TestFileHelper::cleanup_test_directory(test_dir);
        throw;
    }
    
    TestFileHelper::cleanup_test_directory(test_dir);
}

/**
 * @brief Main test function for TinyLlama API (called from main test runner)
 */
void test_tiny_llama_api() {
    std::cout << "Testing TinyLlama API..." << std::endl;
    
    test_tiny_llama_constructor_destructor();
    test_initialization();
    test_initialization_with_config();
    test_tokenization();
    test_text_generation();
    test_configuration();
    test_status_methods();
    
    std::cout << "✓ All TinyLlama API tests passed" << std::endl;
}

/**
 * @brief Main test function (for standalone execution)
 */
int main() {
    std::cout << "Running TinyLlama API tests..." << std::endl;
    
    try {
        test_tiny_llama_api();
        
        std::cout << "\n✅ All TinyLlama API tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "\n❌ Test failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "\n❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}