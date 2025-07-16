#include "tiny_llama/model.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <fstream>

using namespace tiny_llama;

// Forward declarations of test functions
void test_gelu_activation();
void test_forward_pass();
void test_dimension_mismatch();
void test_weight_loading();

int main() {
    std::cout << "Running FFN standalone tests..." << std::endl;
    
    try {
        test_gelu_activation();
        test_forward_pass();
        test_dimension_mismatch();
        test_weight_loading();
        
        std::cout << "All FFN tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

// Test implementations are in test_ffn.cpp