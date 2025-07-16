#include "tiny_llama/tiny_llama.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>

int main() {
    try {
        std::cout << "Tiny Llama C++ Basic Usage Example" << std::endl;
        
        // Create TinyLlama instance
        tiny_llama::TinyLlama llama;
        
        std::cout << "TinyLlama instance created successfully" << std::endl;
        std::cout << "Is ready: " << (llama.is_ready() ? "Yes" : "No") << std::endl;
        
        // Note: Actual initialization and usage will be implemented in later tasks
        std::cout << "Basic usage example completed (stub implementation)" << std::endl;
        
    } catch (const tiny_llama::TinyLlamaException& e) {
        std::cerr << "TinyLlama error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}