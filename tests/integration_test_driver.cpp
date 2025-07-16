#include <iostream>

// Forward declarations
void test_integration();

int main() {
    std::cout << "Running Tiny Llama C++ Integration Tests..." << std::endl;
    
    try {
        test_integration();
        std::cout << "\nIntegration tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Integration tests failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Integration tests failed with unknown exception" << std::endl;
        return 1;
    }
}