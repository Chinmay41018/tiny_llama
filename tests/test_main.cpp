#include <iostream>

// Forward declarations
void test_matrix_operations();
void test_tensor_operations();
void test_exception_handling();
void test_vocabulary();
void test_bpe_tokenizer();
void test_attention();
void test_tiny_llama_api();

int main() {
    std::cout << "Running Tiny Llama C++ Tests..." << std::endl;
    
    int passed = 0;
    int total = 0;
    
    try {
        std::cout << "\n=== Matrix Tests ===" << std::endl;
        test_matrix_operations();
        passed++;
        total++;
    } catch (const std::exception& e) {
        std::cout << "Matrix tests failed: " << e.what() << std::endl;
        total++;
    }
    
    try {
        std::cout << "\n=== Tensor Tests ===" << std::endl;
        test_tensor_operations();
        passed++;
        total++;
    } catch (const std::exception& e) {
        std::cout << "Tensor tests failed: " << e.what() << std::endl;
        total++;
    }
    
    try {
        std::cout << "\n=== Exception Handling Tests ===" << std::endl;
        test_exception_handling();
        passed++;
        total++;
    } catch (const std::exception& e) {
        std::cout << "Exception handling tests failed: " << e.what() << std::endl;
        total++;
    }
    
    try {
        std::cout << "\n=== Vocabulary Tests ===" << std::endl;
        test_vocabulary();
        passed++;
        total++;
    } catch (const std::exception& e) {
        std::cout << "Vocabulary tests failed: " << e.what() << std::endl;
        total++;
    }
    
    try {
        std::cout << "\n=== BPE Tokenizer Tests ===" << std::endl;
        test_bpe_tokenizer();
        passed++;
        total++;
    } catch (const std::exception& e) {
        std::cout << "BPE Tokenizer tests failed: " << e.what() << std::endl;
        total++;
    }
    
    try {
        std::cout << "\n=== Attention Tests ===" << std::endl;
        test_attention();
        passed++;
        total++;
    } catch (const std::exception& e) {
        std::cout << "Attention tests failed: " << e.what() << std::endl;
        total++;
    }
    
    try {
        std::cout << "\n=== TinyLlama API Tests ===" << std::endl;
        test_tiny_llama_api();
        passed++;
        total++;
    } catch (const std::exception& e) {
        std::cout << "TinyLlama API tests failed: " << e.what() << std::endl;
        total++;
    }
    
    std::cout << "\nTests completed: " << passed << "/" << total << " passed" << std::endl;
    
    return (passed == total) ? 0 : 1;
}