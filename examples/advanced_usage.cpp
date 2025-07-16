#include "tiny_llama/tiny_llama.hpp"
#include "tiny_llama/model.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>

int main() {
    try {
        std::cout << "Tiny Llama C++ Advanced Usage Example" << std::endl;
        
        // Create TinyLlama instance
        tiny_llama::TinyLlama llama;
        
        // Create model configuration
        tiny_llama::ModelConfig config;
        config.model_dim = 512;
        config.num_layers = 6;
        config.num_heads = 8;
        config.max_sequence_length = 1024;
        
        std::cout << "Model configuration:" << std::endl;
        std::cout << "  Model dimension: " << config.model_dim << std::endl;
        std::cout << "  Number of layers: " << config.num_layers << std::endl;
        std::cout << "  Number of heads: " << config.num_heads << std::endl;
        std::cout << "  Max sequence length: " << config.max_sequence_length << std::endl;
        
        // Note: Actual model initialization and text generation will be implemented in later tasks
        std::cout << "Advanced usage example completed (stub implementation)" << std::endl;
        
    } catch (const tiny_llama::TinyLlamaException& e) {
        std::cerr << "TinyLlama error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}