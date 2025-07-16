#include "tiny_llama/tiny_llama.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <limits>
#include <cmath>

using namespace tiny_llama;

// Test macros
#define EXPECT_EXCEPTION(exception_type, code) \
    do { \
        bool caught = false; \
        try { \
            code; \
        } catch (const exception_type& e) { \
            caught = true; \
            std::cout << "  Expected exception caught: " << e.what() << std::endl; \
        } catch (const std::exception& e) { \
            std::cout << "  Unexpected exception type: " << e.what() << std::endl; \
            assert(false && "Wrong exception type"); \
        } \
        assert(caught && "Expected exception was not thrown"); \
    } while(0)

// Test parameter validation functions directly by testing configuration methods
// that don't require model initialization
void test_temperature_validation() {
    std::cout << "Testing temperature parameter validation..." << std::endl;
    
    TinyLlama llama;
    
    // Test invalid temperature values
    std::cout << "  Testing invalid temperature values..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(0.0f));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(-1.0f));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(0.005f)); // Below minimum
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(2000.0f)); // Too large
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(std::numeric_limits<float>::infinity()));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(std::numeric_limits<float>::quiet_NaN()));
    
    // Test valid temperature values (should not throw)
    std::cout << "  Testing valid temperature values..." << std::endl;
    try {
        llama.set_temperature(0.01f);
        llama.set_temperature(0.1f);
        llama.set_temperature(1.0f);
        llama.set_temperature(2.0f);
        llama.set_temperature(1000.0f); // At maximum
        std::cout << "  Valid temperature values accepted" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Unexpected exception for valid temperature: " << e.what() << std::endl;
        assert(false && "Valid temperature values should not throw exceptions");
    }
    
    std::cout << "  Temperature validation tests passed!" << std::endl;
}

void test_sequence_length_validation() {
    std::cout << "Testing sequence length parameter validation..." << std::endl;
    
    TinyLlama llama;
    
    // Test invalid sequence length values
    std::cout << "  Testing invalid sequence length values..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(0));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(-1));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(-100));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(200000)); // Too large
    
    // Test that runtime changes are not supported (this is a design decision)
    std::cout << "  Testing runtime sequence length change restriction..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(1024));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(512));
    
    std::cout << "  Sequence length validation tests passed!" << std::endl;
}

void test_boundary_conditions() {
    std::cout << "Testing boundary conditions..." << std::endl;
    
    TinyLlama llama;
    
    // Test temperature boundary values
    std::cout << "  Testing temperature boundaries..." << std::endl;
    try {
        llama.set_temperature(0.01f); // Minimum allowed
        llama.set_temperature(1000.0f); // Maximum allowed
        std::cout << "  Boundary temperature values accepted" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Unexpected exception for boundary temperature: " << e.what() << std::endl;
        assert(false && "Boundary temperature values should be accepted");
    }
    
    // Test just outside boundaries
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(0.009f)); // Just below minimum
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(1000.1f)); // Just above maximum
    
    std::cout << "  Boundary condition tests passed!" << std::endl;
}

void test_string_validation_edge_cases() {
    std::cout << "Testing string validation edge cases..." << std::endl;
    
    TinyLlama llama;
    
    // Test file path validation through initialization methods
    std::cout << "  Testing file path validation..." << std::endl;
    
    // Empty paths
    EXPECT_EXCEPTION(FileIOException, llama.initialize(""));
    EXPECT_EXCEPTION(FileIOException, llama.initialize_with_config("", "", ""));
    
    // Paths with null characters
    std::string null_path = "test\0path";
    EXPECT_EXCEPTION(FileIOException, llama.initialize(null_path));
    
    // Extremely long paths
    std::string long_path(2000000, 'a'); // 2MB string
    EXPECT_EXCEPTION(FileIOException, llama.initialize(long_path));
    
    std::cout << "  String validation edge case tests passed!" << std::endl;
}

void test_numeric_validation_edge_cases() {
    std::cout << "Testing numeric validation edge cases..." << std::endl;
    
    TinyLlama llama;
    
    // Test floating point edge cases
    std::cout << "  Testing floating point edge cases..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(std::numeric_limits<float>::min())); // Very small positive
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(-std::numeric_limits<float>::min())); // Very small negative
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(std::numeric_limits<float>::max())); // Very large
    EXPECT_EXCEPTION(ConfigurationException, llama.set_temperature(-std::numeric_limits<float>::max())); // Very large negative
    
    // Test integer edge cases
    std::cout << "  Testing integer edge cases..." << std::endl;
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(std::numeric_limits<int>::min()));
    EXPECT_EXCEPTION(ConfigurationException, llama.set_max_sequence_length(std::numeric_limits<int>::max()));
    
    std::cout << "  Numeric validation edge case tests passed!" << std::endl;
}

int main() {
    std::cout << "Running Parameter Validation Tests..." << std::endl;
    
    try {
        test_temperature_validation();
        test_sequence_length_validation();
        test_boundary_conditions();
        test_string_validation_edge_cases();
        test_numeric_validation_edge_cases();
        
        std::cout << "\nAll parameter validation tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}