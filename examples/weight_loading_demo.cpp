#include "tiny_llama/model.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>
#include <fstream>

using namespace tiny_llama;

int main() {
    std::cout << "=== Tiny Llama Weight Loading Demo ===" << std::endl;
    
    try {
        // Create a model with default configuration
        std::cout << "Creating model with default configuration..." << std::endl;
        TinyLlamaModel model;
        
        // Display model configuration
        const ModelConfig& config = model.get_config();
        std::cout << "Model Configuration:" << std::endl;
        std::cout << "  Model dimension: " << config.model_dim << std::endl;
        std::cout << "  Number of layers: " << config.num_layers << std::endl;
        std::cout << "  Number of heads: " << config.num_heads << std::endl;
        std::cout << "  FFN hidden dimension: " << config.ffn_hidden_dim << std::endl;
        std::cout << "  Max sequence length: " << config.max_sequence_length << std::endl;
        std::cout << "  Vocabulary size: " << config.vocab_size << std::endl;
        
        // Save model weights to a file
        const std::string weights_file = "demo_weights.bin";
        std::cout << "\nSaving model weights to: " << weights_file << std::endl;
        model.save_model_weights(weights_file);
        
        // Check file size
        std::ifstream file(weights_file, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            size_t file_size = file.tellg();
            std::cout << "Weight file size: " << file_size << " bytes" << std::endl;
            file.close();
        }
        
        // Create a new model and load the weights
        std::cout << "\nCreating new model and loading weights..." << std::endl;
        TinyLlamaModel model2;
        model2.load_model_weights(weights_file);
        std::cout << "Weights loaded successfully!" << std::endl;
        
        // Verify the configuration matches
        const ModelConfig& loaded_config = model2.get_config();
        bool config_matches = (
            loaded_config.model_dim == config.model_dim &&
            loaded_config.num_layers == config.num_layers &&
            loaded_config.num_heads == config.num_heads &&
            loaded_config.ffn_hidden_dim == config.ffn_hidden_dim &&
            loaded_config.max_sequence_length == config.max_sequence_length &&
            loaded_config.vocab_size == config.vocab_size
        );
        
        std::cout << "Configuration verification: " << (config_matches ? "PASSED" : "FAILED") << std::endl;
        
        // Clean up
        std::remove(weights_file.c_str());
        std::cout << "\nDemo completed successfully!" << std::endl;
        
    } catch (const FileIOException& e) {
        std::cerr << "File I/O Error: " << e.what() << std::endl;
        return 1;
    } catch (const ModelException& e) {
        std::cerr << "Model Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}