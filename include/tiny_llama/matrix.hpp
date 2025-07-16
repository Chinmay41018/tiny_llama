#pragma once

#include <vector>
#include <string>
#include <cstddef>

namespace tiny_llama {

/**
 * @brief Template class for 2D matrix operations
 * @tparam T Data type (typically float)
 */
template<typename T>
class Matrix {
private:
    std::vector<T> data_;
    size_t rows_;
    size_t cols_;
    
public:
    /**
     * @brief Construct matrix with given dimensions
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Matrix(size_t rows, size_t cols);
    
    /**
     * @brief Construct matrix from existing data
     * @param rows Number of rows
     * @param cols Number of columns
     * @param data Vector containing matrix data in row-major order
     */
    Matrix(size_t rows, size_t cols, const std::vector<T>& data);
    
    /**
     * @brief Default constructor (empty matrix)
     */
    Matrix() : rows_(0), cols_(0) {}
    
    /**
     * @brief Access element at (row, col)
     * @param row Row index
     * @param col Column index
     * @return Reference to element
     */
    T& operator()(size_t row, size_t col);
    
    /**
     * @brief Access element at (row, col) (const version)
     * @param row Row index
     * @param col Column index
     * @return Const reference to element
     */
    const T& operator()(size_t row, size_t col) const;
    
    /**
     * @brief Matrix multiplication
     * @param other Matrix to multiply with
     * @return Result matrix
     */
    Matrix<T> operator*(const Matrix<T>& other) const;
    
    /**
     * @brief Matrix addition
     * @param other Matrix to add
     * @return Result matrix
     */
    Matrix<T> operator+(const Matrix<T>& other) const;
    
    /**
     * @brief Load matrix from binary file
     * @param filename Path to file
     */
    void load_from_file(const std::string& filename);
    
    /**
     * @brief Save matrix to binary file
     * @param filename Path to file
     */
    void save_to_file(const std::string& filename) const;
    
    /**
     * @brief Get number of rows
     * @return Number of rows
     */
    size_t rows() const { return rows_; }
    
    /**
     * @brief Get number of columns
     * @return Number of columns
     */
    size_t cols() const { return cols_; }
    
    /**
     * @brief Get total number of elements
     * @return Total elements
     */
    size_t size() const { return rows_ * cols_; }
    
    /**
     * @brief Get raw data pointer
     * @return Pointer to data
     */
    T* data() { return data_.data(); }
    
    /**
     * @brief Get raw data pointer (const version)
     * @return Const pointer to data
     */
    const T* data() const { return data_.data(); }
    
    /**
     * @brief Resize matrix
     * @param rows New number of rows
     * @param cols New number of columns
     */
    void resize(size_t rows, size_t cols);
    
    /**
     * @brief Fill matrix with value
     * @param value Value to fill with
     */
    void fill(const T& value);
    
    /**
     * @brief Transpose matrix
     * @return Transposed matrix
     */
    Matrix<T> transpose() const;
};

/**
 * @brief Template class for multi-dimensional tensors
 * @tparam T Data type (typically float)
 */
template<typename T>
class Tensor {
private:
    std::vector<T> data_;
    std::vector<size_t> shape_;
    
    size_t compute_index(const std::vector<size_t>& indices) const;
    
public:
    /**
     * @brief Construct tensor with given shape
     * @param shape Vector specifying dimensions
     */
    explicit Tensor(const std::vector<size_t>& shape);
    
    /**
     * @brief Default constructor (empty tensor)
     */
    Tensor() = default;
    
    /**
     * @brief Access element at given indices
     * @param indices Vector of indices for each dimension
     * @return Reference to element
     */
    T& at(const std::vector<size_t>& indices);
    
    /**
     * @brief Access element at given indices (const version)
     * @param indices Vector of indices for each dimension
     * @return Const reference to element
     */
    const T& at(const std::vector<size_t>& indices) const;
    
    /**
     * @brief Convert to 2D matrix (for 2D tensors)
     * @return Matrix representation
     */
    Matrix<T> to_matrix() const;
    
    /**
     * @brief Get total number of elements
     * @return Total elements
     */
    size_t total_size() const;
    
    /**
     * @brief Get tensor shape
     * @return Vector of dimensions
     */
    const std::vector<size_t>& shape() const { return shape_; }
    
    /**
     * @brief Get number of dimensions
     * @return Number of dimensions
     */
    size_t ndim() const { return shape_.size(); }
    
    /**
     * @brief Resize tensor
     * @param shape New shape
     */
    void resize(const std::vector<size_t>& shape);
    
    /**
     * @brief Fill tensor with value
     * @param value Value to fill with
     */
    void fill(const T& value);
};

} // namespace tiny_llama