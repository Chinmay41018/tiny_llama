#include <iostream>

// Forward declarations of test functions
void test_transformer_block_init();
void test_transformer_block_forward();
void test_residual_connections();
void test_null_mask();

int main() {
    try {
        std::cout << "Running transformer block tests..." << std::endl;
        
        test_transformer_block_init();
        test_transformer_block_forward();
        test_residual_connections();
        test_null_mask();
        
        std::cout << "All transformer block tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}