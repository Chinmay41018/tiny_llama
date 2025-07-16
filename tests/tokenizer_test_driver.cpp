#include <iostream>

// Forward declaration
void test_bpe_tokenizer();

int main() {
    std::cout << "Running BPE Tokenizer Tests..." << std::endl;
    
    try {
        test_bpe_tokenizer();
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}