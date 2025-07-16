#include "tiny_llama/matrix.hpp"
#include "tiny_llama/exceptions.hpp"
#include <stdexcept>
#include <fstream>
#include <algorithm>

namespace tiny_llama {

// Matrix template implementation
template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
    data_.resize(rows * cols);
}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const std::vector<T>& data) 
    : data_(data), rows_(rows), cols_(cols) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size does not match matrix dimensions");
    }
}

template<typename T>
T& Matrix<T>::operator()(size_t row, size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[row * cols_ + col];
}

template<typename T>
const T& Matrix<T>::operator()(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[row * cols_ + col];
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix<T> result(rows_, other.cols_);
    
    for (size_t i =0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            T sum = T{0};
            for (size_t k = 0; k < cols_; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix<T> result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    
    return result;
}

template<typename T>
void Matrix<T>::load_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw FileIOException("Cannot open file for reading: " + filename);
    }
    
    // Read dimensions
    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    
    if (file.fail()) {
        throw FileIOException("Failed to read matrix dimensions from file: " + filename);
    }
    
    // Resize matrix
    resize(rows, cols);
    
    // Read data
    file.read(reinterpret_cast<char*>(data_.data()), data_.size() * sizeof(T));
    
    if (file.fail()) {
        throw FileIOException("Failed to read matrix data from file: " + filename);
    }
    
    file.close();
}

template<typename T>
void Matrix<T>::save_to_file(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw FileIOException("Cannot open file for writing: " + filename);
    }
    
    // Write dimensions
    file.write(reinterpret_cast<const char*>(&rows_), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(&cols_), sizeof(size_t));
    
    if (file.fail()) {
        throw FileIOException("Failed to write matrix dimensions to file: " + filename);
    }
    
    // Write data
    file.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(T));
    
    if (file.fail()) {
        throw FileIOException("Failed to write matrix data to file: " + filename);
    }
    
    file.close();
}

template<typename T>
void Matrix<T>::resize(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;
    data_.resize(rows * cols);
}

template<typename T>
void Matrix<T>::fill(const T& value) {
    std::fill(data_.begin(), data_.end(), value);
}

template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols_, rows_);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    
    return result;
}

// Tensor template implementation
template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    data_.resize(total);
}

template<typename T>
T& Tensor<T>::at(const std::vector<size_t>& indices) {
    size_t index = compute_index(indices);
    return data_[index];
}

template<typename T>
const T& Tensor<T>::at(const std::vector<size_t>& indices) const {
    size_t index = compute_index(indices);
    return data_[index];
}

template<typename T>
size_t Tensor<T>::compute_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices does not match tensor dimensions");
    }
    
    size_t index = 0;
    size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of bounds");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }
    return index;
}

template<typename T>
Matrix<T> Tensor<T>::to_matrix() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("Can only convert 2D tensors to matrices");
    }
    
    size_t rows = shape_[0];
    size_t cols = shape_[1];
    
    return Matrix<T>(rows, cols, data_);
}

template<typename T>
size_t Tensor<T>::total_size() const {
    size_t total = 1;
    for (size_t dim : shape_) {
        total *= dim;
    }
    return total;
}

template<typename T>
void Tensor<T>::resize(const std::vector<size_t>& shape) {
    shape_ = shape;
    size_t total = total_size();
    data_.resize(total);
}

template<typename T>
void Tensor<T>::fill(const T& value) {
    std::fill(data_.begin(), data_.end(), value);
}

// Explicit template instantiations
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<int>;

template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;

} // namespace tiny_llama