#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <random>
#include <unistd.h> // For getcwd

// Simple program to create test data for integration tests
int main() {
    // Get the current working directory
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        std::cerr << "Failed to get current working directory" << std::endl;
        return 1;
    }
    std::string current_dir(cwd);
    std::cout << "Current directory: " << current_dir << std::endl;
    
    // Create a binary weights file with absolute path
    std::string weights_path = current_dir + "/tiny_llama_cpp/data/weights.bin";
    std::ofstream weights_file(weights_path, std::ios::binary);
    if (!weights_file) {
        std::cerr << "Failed to create weights file at: " << weights_path << std::endl;
        return 1;
    }

    // Magic number "TLLM"
    uint32_t magic = 0x544C4C4D;
    weights_file.write(reinterpret_cast<char*>(&magic), sizeof(magic));

    // Version
    uint32_t version = 1;
    weights_file.write(reinterpret_cast<char*>(&version), sizeof(version));

    // Model configuration (match default values in ModelConfig)
    int model_dim = 64;
    int num_layers = 2;
    int num_heads = 2;
    int ffn_hidden_dim = 128;
    int max_sequence_length = 128;
    int vocab_size = 100;  // Match with vocab.txt
    float dropout_rate = 0.1f;

    weights_file.write(reinterpret_cast<char*>(&model_dim), sizeof(model_dim));
    weights_file.write(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    weights_file.write(reinterpret_cast<char*>(&num_heads), sizeof(num_heads));
    weights_file.write(reinterpret_cast<char*>(&ffn_hidden_dim), sizeof(ffn_hidden_dim));
    weights_file.write(reinterpret_cast<char*>(&max_sequence_length), sizeof(max_sequence_length));
    weights_file.write(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    weights_file.write(reinterpret_cast<char*>(&dropout_rate), sizeof(dropout_rate));

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    // Helper function to write a matrix
    auto write_matrix = [&](size_t rows, size_t cols) {
        weights_file.write(reinterpret_cast<char*>(&rows), sizeof(rows));
        weights_file.write(reinterpret_cast<char*>(&cols), sizeof(cols));
        
        for (size_t i = 0; i < rows * cols; i++) {
            float val = dist(gen);
            weights_file.write(reinterpret_cast<char*>(&val), sizeof(val));
        }
    };

    // Helper function to write a vector
    auto write_vector = [&](size_t size) {
        weights_file.write(reinterpret_cast<char*>(&size), sizeof(size));
        
        for (size_t i = 0; i < size; i++) {
            float val = dist(gen);
            weights_file.write(reinterpret_cast<char*>(&val), sizeof(val));
        }
    };

    // Write embedding weights
    write_matrix(vocab_size, model_dim);
    
    // Write position embeddings
    write_matrix(max_sequence_length, model_dim);
    
    // Write transformer blocks
    for (int layer = 0; layer < num_layers; layer++) {
        // Attention weights
        write_matrix(model_dim, model_dim); // query
        write_matrix(model_dim, model_dim); // key
        write_matrix(model_dim, model_dim); // value
        write_matrix(model_dim, model_dim); // output
        
        // FFN weights
        write_matrix(model_dim, ffn_hidden_dim); // linear1
        write_vector(ffn_hidden_dim); // linear1 bias
        write_matrix(ffn_hidden_dim, model_dim); // linear2
        write_vector(model_dim); // linear2 bias
        
        // Layer norm weights
        write_vector(model_dim); // layer norm 1 weights
        write_vector(model_dim); // layer norm 1 bias
        write_vector(model_dim); // layer norm 2 weights
        write_vector(model_dim); // layer norm 2 bias
    }
    
    // Write output projection
    write_matrix(model_dim, vocab_size);
    
    weights_file.close();
    std::cout << "Created test weights file" << std::endl;

    // Create a more comprehensive vocabulary file
    std::string vocab_path = current_dir + "/tiny_llama_cpp/data/vocab.txt";
    std::ofstream vocab_file(vocab_path);
    if (!vocab_file) {
        std::cerr << "Failed to create vocabulary file at: " << vocab_path << std::endl;
        return 1;
    }

    // Special tokens
    vocab_file << "<unk>\n<pad>\n<bos>\n<eos>\n";

    // Basic ASCII characters
    for (int i = 32; i < 127; i++) {
        vocab_file << static_cast<char>(i) << "\n";
    }

    // Common words
    std::vector<std::string> common_words = {
        "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
        "with", "as", "was", "on", "be", "at", "by", "this", "have", "from",
        "or", "had", "an", "but", "are", "not", "they", "which", "you", "one"
    };

    for (const auto& word : common_words) {
        vocab_file << word << "\n";
    }

    vocab_file.close();
    std::cout << "Created test vocabulary file" << std::endl;

    // Create a simple BPE merges file
    std::string merges_path = current_dir + "/tiny_llama_cpp/data/merges.txt";
    std::ofstream merges_file(merges_path);
    if (!merges_file) {
        std::cerr << "Failed to create merges file at: " << merges_path << std::endl;
        return 1;
    }

    merges_file << "#version: 0.2\n";
    
    // Add some basic merges
    std::vector<std::string> merges = {
        "t h", "th e", "the",
        "a n", "an d", "and",
        "i n", "in",
        "i s", "is",
        "t o", "to",
        "f o", "fo r", "for",
        "i t", "it"
    };

    for (const auto& merge : merges) {
        merges_file << merge << "\n";
    }

    merges_file.close();
    std::cout << "Created test merges file" << std::endl;

    return 0;
}