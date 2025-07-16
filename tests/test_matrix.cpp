#include "tiny_llama/matrix.hpp"
#include "tiny_llama/exceptions.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace tiny_llama;

void test_matrix_construction() {
    std::cout << "Testing matrix construction..." << std::endl;
    
    // Test default constructor
    Matrix<float> empty_matrix;
    assert(empty_matrix.rows() == 0);
    assert(empty_matrix.cols() == 0);
    assert(empty_matrix.size() == 0);
    
    // Test parameterized constructor
    Matrix<float> matrix(3, 4);
    assert(matrix.rows() == 3);
    assert(matrix.cols() == 4);
    assert(matrix.size() == 12);
    
    // Test constructor with data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    Matrix<float> matrix_with_data(2, 2, data);
    assert(matrix_with_data.rows() == 2);
    assert(matrix_with_data.cols() == 2);
    assert(matrix_with_data(0, 0) == 1.0f);
    assert(matrix_with_data(0, 1) == 2.0f);
    assert(matrix_with_data(1, 0) == 3.0f);
    assert(matrix_with_data(1, 1) == 4.0f);
    
    std::cout << "Matrix construction tests passed!" << std::endl;
}

void test_matrix_indexing() {
    std::cout << "Testing matrix indexing..." << std::endl;
    
    Matrix<float> matrix(2, 3);
    
    // Test setting values
    matrix(0, 0) = 1.0f;
    matrix(0, 1) = 2.0f;
    matrix(0, 2) = 3.0f;
    matrix(1, 0) = 4.0f;
    matrix(1, 1) = 5.0f;
    matrix(1, 2) = 6.0f;
    
    // Test getting values
    assert(matrix(0, 0) == 1.0f);
    assert(matrix(0, 1) == 2.0f);
    assert(matrix(0, 2) == 3.0f);
    assert(matrix(1, 0) == 4.0f);
    assert(matrix(1, 1) == 5.0f);
    assert(matrix(1, 2) == 6.0f);
    
    // Test bounds checking
    bool exception_thrown = false;
    try {
        matrix(2, 0) = 7.0f;  // Should throw
    } catch (const std::out_of_range&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    exception_thrown = false;
    try {
        matrix(0, 3) = 7.0f;  // Should throw
    } catch (const std::out_of_range&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Matrix indexing tests passed!" << std::endl;
}

void test_matrix_multiplication() {
    std::cout << "Testing matrix multiplication..." << std::endl;
    
    // Test 2x3 * 3x2 = 2x2
    Matrix<float> a(2, 3);
    a(0, 0) = 1.0f; a(0, 1) = 2.0f; a(0, 2) = 3.0f;
    a(1, 0) = 4.0f; a(1, 1) = 5.0f; a(1, 2) = 6.0f;
    
    Matrix<float> b(3, 2);
    b(0, 0) = 7.0f; b(0, 1) = 8.0f;
    b(1, 0) = 9.0f; b(1, 1) = 10.0f;
    b(2, 0) = 11.0f; b(2, 1) = 12.0f;
    
    Matrix<float> c = a * b;
    
    assert(c.rows() == 2);
    assert(c.cols() == 2);
    
    // Expected result:
    // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
    // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
    assert(std::abs(c(0, 0) - 58.0f) < 1e-6f);
       assert(std::abs(c(0, 1) - 64.0f) < 1e-6f);
    assert(std::abs(c(1, 0) - 139.0f) < 1e-6f);
    assert(std::abs(c(1, 1) - 154.0f) < 1e-6f);
    
    // Test dimension mismatch
    Matrix<float> d(2, 2);
    bool exception_thrown = false;
    try {
        Matrix<float> result = a * d;  // Should throw (2x3 * 2x2 invalid)
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Matrix multiplication tests passed!" << std::endl;
}

void test_matrix_addition() {
    std::cout << "Testing matrix addition..." << std::endl;
    
    Matrix<float> a(2, 2);
    a(0, 0) = 1.0f; a(0, 1) = 2.0f;
    a(1, 0) = 3.0f; a(1, 1) = 4.0f;
    
    Matrix<float> b(2, 2);
    b(0, 0) = 5.0f; b(0, 1) = 6.0f;
    b(1, 0) = 7.0f; b(1, 1) = 8.0f;
    
    Matrix<float> c = a + b;
    
    assert(c.rows() == 2);
    assert(c.cols() == 2);
    assert(c(0, 0) == 6.0f);
    assert(c(0, 1) == 8.0f);
    assert(c(1, 0) == 10.0f);
    assert(c(1, 1) == 12.0f);
    
    // Test dimension mismatch
    Matrix<float> d(2, 3);
    bool exception_thrown = false;
    try {
        Matrix<float> result = a + d;  // Should throw
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Matrix addition tests passed!" << std::endl;
}

void test_matrix_transpose() {
    std::cout << "Testing matrix transpose..." << std::endl;
    
    Matrix<float> a(2, 3);
    a(0, 0) = 1.0f; a(0, 1) = 2.0f; a(0, 2) = 3.0f;
    a(1, 0) = 4.0f; a(1, 1) = 5.0f; a(1, 2) = 6.0f;
    
    Matrix<float> b = a.transpose();
    
    assert(b.rows() == 3);
    assert(b.cols() == 2);
    assert(b(0, 0) == 1.0f); assert(b(0, 1) == 4.0f);
    assert(b(1, 0) == 2.0f); assert(b(1, 1) == 5.0f);
    assert(b(2, 0) == 3.0f); assert(b(2, 1) == 6.0f);
    
    std::cout << "Matrix transpose tests passed!" << std::endl;
}

void test_matrix_file_io() {
    std::cout << "Testing matrix file I/O..." << std::endl;
    
    // Create test matrix
    Matrix<float> original(2, 3);
    original(0, 0) = 1.5f; original(0, 1) = 2.5f; original(0, 2) = 3.5f;
    original(1, 0) = 4.5f; original(1, 1) = 5.5f; original(1, 2) = 6.5f;
    
    // Save to file
    std::string filename = "test_matrix.bin";
    original.save_to_file(filename);
    
    // Load from file
    Matrix<float> loaded;
    loaded.load_from_file(filename);
    
    // Verify loaded matrix
    assert(loaded.rows() == 2);
    assert(loaded.cols() == 3);
    assert(std::abs(loaded(0, 0) - 1.5f) < 1e-6f);
    assert(std::abs(loaded(0, 1) - 2.5f) < 1e-6f);
    assert(std::abs(loaded(0, 2) - 3.5f) < 1e-6f);
    assert(std::abs(loaded(1, 0) - 4.5f) < 1e-6f);
    assert(std::abs(loaded(1, 1) - 5.5f) < 1e-6f);
    assert(std::abs(loaded(1, 2) - 6.5f) < 1e-6f);
    
    // Clean up
    std::remove(filename.c_str());
    
    // Test file not found
    bool exception_thrown = false;
    try {
        Matrix<float> test;
        test.load_from_file("nonexistent_file.bin");
    } catch (const FileIOException&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Matrix file I/O tests passed!" << std::endl;
}

void test_matrix_utilities() {
    std::cout << "Testing matrix utilities..." << std::endl;
    
    Matrix<float> matrix(3, 4);
    
    // Test fill
    matrix.fill(7.5f);
    for (size_t i = 0; i < matrix.rows(); ++i) {
        for (size_t j = 0; j < matrix.cols(); ++j) {
            assert(matrix(i, j) == 7.5f);
        }
    }
    
    // Test resize
    matrix.resize(2, 5);
    assert(matrix.rows() == 2);
    assert(matrix.cols() == 5);
    assert(matrix.size() == 10);
    
    // Test data access
    matrix.fill(3.14f);
    const float* data_ptr = matrix.data();
    assert(data_ptr != nullptr);
    assert(data_ptr[0] == 3.14f);
    
    std::cout << "Matrix utilities tests passed!" << std::endl;
}

void test_matrix_operations() {
    test_matrix_construction();
    test_matrix_indexing();
    test_matrix_multiplication();
    test_matrix_addition();
    test_matrix_transpose();
    test_matrix_file_io();
    test_matrix_utilities();
    
    std::cout << "All Matrix tests passed!" << std::endl;
}

void test_tensor_construction() {
    std::cout << "Testing tensor construction..." << std::endl;
    
    // Test default constructor
    Tensor<float> empty_tensor;
    assert(empty_tensor.ndim() == 0);
    assert(empty_tensor.total_size() == 1);  // Empty shape means scalar
    
    // Test 1D tensor
    Tensor<float> tensor_1d({5});
    assert(tensor_1d.ndim() == 1);
    assert(tensor_1d.shape()[0] == 5);
    assert(tensor_1d.total_size() == 5);
    
    // Test 2D tensor
    Tensor<float> tensor_2d({3, 4});
    assert(tensor_2d.ndim() == 2);
    assert(tensor_2d.shape()[0] == 3);
    assert(tensor_2d.shape()[1] == 4);
    assert(tensor_2d.total_size() == 12);
    
    // Test 3D tensor
    Tensor<float> tensor_3d({2, 3, 4});
    assert(tensor_3d.ndim() == 3);
    assert(tensor_3d.shape()[0] == 2);
    assert(tensor_3d.shape()[1] == 3);
    assert(tensor_3d.shape()[2] == 4);
    assert(tensor_3d.total_size() == 24);
    
    // Test 4D tensor
    Tensor<float> tensor_4d({2, 3, 4, 5});
    assert(tensor_4d.ndim() == 4);
    assert(tensor_4d.total_size() == 120);
    
    std::cout << "Tensor construction tests passed!" << std::endl;
}

void test_tensor_indexing() {
    std::cout << "Testing tensor indexing..." << std::endl;
    
    // Test 2D tensor indexing
    Tensor<float> tensor_2d({3, 4});
    
    // Set values
    tensor_2d.at({0, 0}) = 1.0f;
    tensor_2d.at({0, 1}) = 2.0f;
    tensor_2d.at({1, 2}) = 3.5f;
    tensor_2d.at({2, 3}) = 4.7f;
    
    // Get values
    assert(tensor_2d.at({0, 0}) == 1.0f);
    assert(tensor_2d.at({0, 1}) == 2.0f);
    assert(tensor_2d.at({1, 2}) == 3.5f);
    assert(tensor_2d.at({2, 3}) == 4.7f);
    
    // Test 3D tensor indexing
    Tensor<float> tensor_3d({2, 3, 4});
    tensor_3d.at({0, 1, 2}) = 5.5f;
    tensor_3d.at({1, 2, 3}) = 6.6f;
    
    assert(tensor_3d.at({0, 1, 2}) == 5.5f);
    assert(tensor_3d.at({1, 2, 3}) == 6.6f);
    
    // Test bounds checking
    bool exception_thrown = false;
    try {
        tensor_2d.at({3, 0}) = 7.0f;  // Should throw (row out of bounds)
    } catch (const std::out_of_range&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    exception_thrown = false;
    try {
        tensor_2d.at({0, 4}) = 7.0f;  // Should throw (col out of bounds)
    } catch (const std::out_of_range&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    exception_thrown = false;
    try {
        tensor_2d.at({0}) = 7.0f;  // Should throw (wrong number of indices)
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Tensor indexing tests passed!" << std::endl;
}

void test_tensor_to_matrix_conversion() {
    std::cout << "Testing tensor to matrix conversion..." << std::endl;
    
    // Test 2D tensor to matrix conversion
    Tensor<float> tensor_2d({2, 3});
    tensor_2d.at({0, 0}) = 1.0f; tensor_2d.at({0, 1}) = 2.0f; tensor_2d.at({0, 2}) = 3.0f;
    tensor_2d.at({1, 0}) = 4.0f; tensor_2d.at({1, 1}) = 5.0f; tensor_2d.at({1, 2}) = 6.0f;
    
    Matrix<float> matrix = tensor_2d.to_matrix();
    
    assert(matrix.rows() == 2);
    assert(matrix.cols() == 3);
    assert(matrix(0, 0) == 1.0f);
    assert(matrix(0, 1) == 2.0f);
    assert(matrix(0, 2) == 3.0f);
    assert(matrix(1, 0) == 4.0f);
    assert(matrix(1, 1) == 5.0f);
    assert(matrix(1, 2) == 6.0f);
    
    // Test conversion of non-2D tensor should throw
    Tensor<float> tensor_3d({2, 3, 4});
    bool exception_thrown = false;
    try {
        Matrix<float> invalid_matrix = tensor_3d.to_matrix();
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    // Test 1D tensor conversion should also throw
    Tensor<float> tensor_1d({5});
    exception_thrown = false;
    try {
        Matrix<float> invalid_matrix = tensor_1d.to_matrix();
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }
    assert(exception_thrown);
    
    std::cout << "Tensor to matrix conversion tests passed!" << std::endl;
}

void test_tensor_utilities() {
    std::cout << "Testing tensor utilities..." << std::endl;
    
    // Test fill
    Tensor<float> tensor({2, 3, 4});
    tensor.fill(7.5f);
    
    // Check all elements are filled
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                assert(tensor.at({i, j, k}) == 7.5f);
            }
        }
    }
    
    // Test resize
    tensor.resize({3, 2});
    assert(tensor.ndim() == 2);
    assert(tensor.shape()[0] == 3);
    assert(tensor.shape()[1] == 2);
    assert(tensor.total_size() == 6);
    
    // Test total_size calculation for various shapes
    Tensor<float> tensor_1d({10});
    assert(tensor_1d.total_size() == 10);
    
    Tensor<float> tensor_2d({5, 6});
    assert(tensor_2d.total_size() == 30);
    
    Tensor<float> tensor_4d({2, 3, 4, 5});
    assert(tensor_4d.total_size() == 120);
    
    std::cout << "Tensor utilities tests passed!" << std::endl;
}

void test_tensor_memory_management() {
    std::cout << "Testing tensor memory management (RAII)..." << std::endl;
    
    // Test that tensors properly manage memory through RAII
    {
        Tensor<float> tensor({1000, 1000});  // Large tensor
        tensor.fill(3.14f);
        
        // Verify data is accessible
        assert(tensor.at({0, 0}) == 3.14f);
        assert(tensor.at({999, 999}) == 3.14f);
        
        // Test resize doesn't leak memory
        tensor.resize({500, 500});
        tensor.fill(2.71f);
        assert(tensor.at({0, 0}) == 2.71f);
        assert(tensor.at({499, 499}) == 2.71f);
        
    } // Tensor should be automatically destroyed here
    
    // Test copy semantics work correctly
    Tensor<float> original({3, 3});
    original.fill(1.0f);
    
    Tensor<float> copy = original;  // Copy constructor
    copy.at({1, 1}) = 2.0f;
    
    // Original should be unchanged
    assert(original.at({1, 1}) == 1.0f);
    assert(copy.at({1, 1}) == 2.0f);
    
    std::cout << "Tensor memory management tests passed!" << std::endl;
}

void test_tensor_edge_cases() {
    std::cout << "Testing tensor edge cases..." << std::endl;
    
    // Test single element tensor
    Tensor<float> scalar({1});
    scalar.at({0}) = 42.0f;
    assert(scalar.at({0}) == 42.0f);
    assert(scalar.total_size() == 1);
    
    // Test tensor with zero dimension (should still work)
    Tensor<float> tensor_with_zero({0, 5});
    assert(tensor_with_zero.total_size() == 0);
    
    // Test very high dimensional tensor
    Tensor<float> high_dim({2, 2, 2, 2, 2});  // 5D tensor
    assert(high_dim.ndim() == 5);
    assert(high_dim.total_size() == 32);
    
    high_dim.at({1, 1, 1, 1, 1}) = 99.0f;
    assert(high_dim.at({1, 1, 1, 1, 1}) == 99.0f);
    
    std::cout << "Tensor edge cases tests passed!" << std::endl;
}

void test_tensor_operations() {
    test_tensor_construction();
    test_tensor_indexing();
    test_tensor_to_matrix_conversion();
    test_tensor_utilities();
    test_tensor_memory_management();
    test_tensor_edge_cases();
    
    std::cout << "All Tensor tests passed!" << std::endl;
}