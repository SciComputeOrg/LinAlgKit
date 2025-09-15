#ifndef MATRIXLIB_H
#define MATRIXLIB_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <initializer_list>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <type_traits>

namespace matrixlib {

/**
 * @class Matrix
 * @brief A template class for matrix operations
 * 
 * @tparam T Type of elements in the matrix (e.g., int, float, double)
 */
template <typename T>
class Matrix {
    static_assert(std::is_arithmetic<T>::value, "Matrix can only be instantiated with arithmetic types");

private:
    std::vector<std::vector<T>> data;
    size_t rows;
    size_t cols;

public:
    // Constructors
    Matrix() : rows(0), cols(0) {}
    
    /**
     * @brief Construct a matrix with given dimensions
     * @param rows Number of rows
     * @param cols Number of columns
     * @param value Initial value for all elements (default: 0)
     */
    Matrix(size_t rows, size_t cols, const T& value = T());
    
    /**
     * @brief Construct from initializer list
     * @param init 2D initializer list
     */
    Matrix(std::initializer_list<std::initializer_list<T>> init);
    
    // Copy and move constructors
    Matrix(const Matrix& other) = default;
    Matrix(Matrix&& other) noexcept;
    
    // Assignment operators
    Matrix& operator=(const Matrix& other) = default;
    Matrix& operator=(Matrix&& other) noexcept;
    
    // Access operators
    std::vector<T>& operator[](size_t index);
    const std::vector<T>& operator[](size_t index) const;
    
    // Matrix operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(const T& scalar) const;
    
    // Compound assignment operators
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(const Matrix& other);
    Matrix& operator*=(const T& scalar);
    
    // Comparison operators
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;
    
    // Matrix operations
    Matrix transpose() const;
    T trace() const;
    // Optimized determinant using Bareiss algorithm (fraction-free LU), O(n^3)
    T determinant() const;
    // Naive recursive determinant (Laplace expansion), for testing/small n
    T determinant_naive() const;
    Matrix inverse() const;
    
    // Utility functions
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    bool isSquare() const { return rows == cols; }
    bool isEmpty() const { return rows == 0 || cols == 0; }
    
    // Static factory methods
    static Matrix identity(size_t size);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    
    // Input/output
    template <typename U>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<U>& matrix);
    
    template <typename U>
    friend std::istream& operator>>(std::istream& is, Matrix<U>& matrix);
};

// Non-member operator overloads
template <typename T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& matrix);

// Type aliases for common matrix types
using Matrixi = Matrix<int>;
using Matrixf = Matrix<float>;
using Matrixd = Matrix<double>;

} // namespace matrixlib

// Include implementation
#include "matrixlib_impl.h"

#endif // MATRIXLIB_H
