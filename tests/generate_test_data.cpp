#include "tiny_llama/model.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <cstdint>

using namespace tiny_llama;

// Helper function to create a simple vocabulary file for testing
void create_test_vocabulary(const std::string& filepath, int vocab_size = 1000) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to create vocabulary file: " << filepath << std::endl;
        return;
    }
    
    // Add special tokens
    file << "<unk>\n";
    file << "<pad>\n";
    file << "<bos>\n";
    file << "<eos>\n";
    
    // Add basic ASCII characters
    for (int i = 32; i < 127; i++) {
        file << static_cast<char>(i) << "\n";
    }
    
    // Add some multi-character tokens
    const std::vector<std::string> common_tokens = {
        "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
        "with", "as", "was", "on", "be", "at", "by", "this", "have", "from",
        "or", "had", "an", "but", "are", "not", "they", "which", "you", "one",
        "were", "all", "we", "when", "there", "can", "who", "been", "has", "their",
        "if", "would", "will", "what", "about", "so", "no", "out", "up", "into"
    };
    
    for (const auto& token : common_tokens) {
        file << token << "\n";
    }
    
    // Fill the rest with random character combinations
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> char_dist(97, 122); // a-z
    std::uniform_int_distribution<> len_dist(2, 5);
    
    int remaining = vocab_size - 4 - 95 - common_tokens.size();
    for (int i = 0; i < remaining; i++) {
        int length = len_dist(gen);
        std::string token;
        for (int j = 0; j < length; j++) {
            token += static_cast<char>(char_dist(gen));
        }
        file << token << "\n";
    }
    
    file.close();
    std::cout << "Created test vocabulary file with " << vocab_size << " tokens" << std::endl;
}

// Helper function to create a simple BPE merges file for testing
void create_test_merges(const std::string& filepath, int num_merges = 500) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to create merges file: " << filepath << std::endl;
        return;
    }
    
    // Add version header
    file << "#version: 0.2\n";
    
    // Generate random character pairs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> char_dist(97, 122); // a-z
    
    for (int i = 0; i < num_merges; i++) {
        char c1 = static_cast<char>(char_dist(gen));
        char c2 = static_cast<char>(char_dist(gen));
        file << c1 << " " << c2 << "\n";
    }
    
    file.close();
    std::cout << "Created test merges file with " << num_merges << " merges" << std::endl;
}

// Helper function to create a simple model weights file for testing
void create_test_weights(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to create weights file: " << filepath << std::endl;
        return;
    }
    
    // Use a small model configuration for testing
    ModelConfig config;
    config.model_dim = 64;
    config.num_layers = 2;
    config.num_heads = 2;
    config.ffn_hidden_dim = 128;
    config.max_sequence_length = 128;
    config.vocab_size = 1000;
    config.dropout_rate = 0.1f;
    
    // Write magic number "TLLM"
    uint32_t magic = 0x544C4C4D;
    file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
    
    // Write version
    uint32_t version = 1;
    file.write(reinterpret_cast<char*>(&version), sizeof(version));
    
    // Write model configuration
    file.write(reinterpret_cast<char*>(&config.model_dim), sizeof(config.model_dim));
    file.write(reinterpret_cast<char*>(&config.num_layers), sizeof(config.num_layers));
    file.write(reinterpret_cast<char*>(&config.num_heads), sizeof(config.num_heads));
    file.write(reinterpret_cast<char*>(&config.ffn_hidden_dim), sizeof(config.ffn_hidden_dim));
    file.write(reinterpret_cast<char*>(&config.max_sequence_length), sizeof(config.max_sequence_length));
    file.write(reinterpret_cast<char*>(&config.vocab_size), sizeof(config.vocab_size));
    file.write(reinterpret_cast<char*>(&config.dropout_rate), sizeof(config.dropout_rate));
    
    // Random number generator for weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    // Helper function to write a matrix
    auto write_matrix = [&](size_t rows, size_t cols) {
        file.write(reinterpret_cast<char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<char*>(&cols), sizeof(cols));
        
        for (size_t i = 0; i < rows * cols; i++) {
            float val = dist(gen);
            file.write(reinterpret_cast<char*>(&val), sizeof(val));
        }
    };
    
    // Helper function to write a vector
    auto write_vector = [&](size_t size) {
        file.write(reinterpret_cast<char*>(&size), sizeof(size));
        
        for (size_t i = 0; i < size; i++) {
            float val = dist(gen);
            file.write(reinterpret_cast<char*>(&val), sizeof(val));
        }
    };
    
    // Write embedding weights
    write_matrix(config.vocab_size, config.model_dim);
    
    // Write position embeddings
    write_matrix(config.max_sequence_length, config.model_dim);
    
    // Write transformer blocks
    for (int layer = 0; layer < config.num_layers; layer++) {
        // Attention weights
        write_matrix(config.model_dim, config.model_dim); // query
        write_matrix(config.model_dim, config.model_dim); // key
        write_matrix(config.model_dim, config.model_dim); // value
        write_matrix(config.model_dim, config.model_dim); // output
        
        // FFN weights
        write_matrix(config.model_dim, config.ffn_hidden_dim); // linear1
        write_vector(config.ffn_hidden_dim); // linear1 bias
        write_matrix(config.ffn_hidden_dim, config.model_dim); // linear2
        write_vector(config.model_dim); // linear2 bias
        
        // Layer norm weights
        write_vector(config.model_dim); // layer norm 1 weights
        write_vector(config.model_dim); // layer norm 1 bias
        write_vector(config.model_dim); // layer norm 2 weights
        write_vector(config.model_dim); // layer norm 2 bias
    }
    
    // Write output projection
    write_matrix(config.model_dim, config.vocab_size);
    
    file.close();
    std::cout << "Created test weights file with configuration:" << std::endl;
    std::cout << "  - Model dimension: " << config.model_dim << std::endl;
    std::cout << "  - Layers: " << config.num_layers << std::endl;
    std::cout << "  - Heads: " << config.num_heads << std::endl;
    std::cout << "  - Vocabulary size: " << config.vocab_size << std::endl;
}

int main(int argc, char** argv) {
    std::string data_dir = "../data";
    
    if (argc > 1) {
        data_dir = argv[1];
    }
    
    std::cout << "Generating test data in directory: " << data_dir << std::endl;
    
    create_test_vocabulary(data_dir + "/vocab.txt");
    create_test_merges(data_dir + "/merges.txt");
    create_test_weights(data_dir + "/weights.bin");
    
    std::cout << "Test data generation complete!" << std::endl;
    return 0;
}