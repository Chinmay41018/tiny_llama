#include <iostream>

// Forward declaration
void test_vocabulary();

int main() {
    std::cout << "Running Vocabulary Tests..." << std::endl;
    
    try {
        test_vocabulary();
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}