#include "matrixlib.h"

namespace matrixlib {

// Implementation of Matrix class methods

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T& value)
    : rows(rows), cols(cols), data(rows * cols, value) {}

template <typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> init) {
    rows = init.size();
    if (rows == 0) {
        cols = 0;
        return;
    }
    cols = init.begin()->size();
    data.assign(rows * cols, T());
    size_t i = 0;
    for (const auto& row : init) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        size_t j = 0;
        for (const auto& v : row) {
            at(i, j) = v;
            ++j;
        }
        ++i;
    }
}

template <typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept
    : data(std::move(other.data)), rows(other.rows), cols(other.cols) {
    other.rows = 0;
    other.cols = 0;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        data = std::move(other.data);
        rows = other.rows;
        cols = other.cols;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

// operator[] implemented via RowProxy in header for contiguous layout

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i,j) = at(i,j) + other.at(i,j);
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i,j) = at(i,j) - other.at(i,j);
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument(
            "Number of columns in first matrix must equal number of rows in second matrix for multiplication");
    }
    
    Matrix result(rows, other.cols, 0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t k = 0; k < other.cols; ++k) {
            T sum = 0;
            for (size_t j = 0; j < cols; ++j) {
                sum += at(i,j) * other.at(j,k);
            }
            result.at(i,k) = sum;
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i,j) = at(i,j) * scalar;
        }
    }
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix& other) {
    *this = *this + other;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix& other) {
    *this = *this - other;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix& other) {
    *this = *this * other;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const T& scalar) {
    *this = *this * scalar;
    return *this;
}

template <typename T>
bool Matrix<T>::operator==(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        return false;
    }
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (data[i][j] != other.data[i][j]) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool Matrix<T>::operator!=(const Matrix& other) const {
    return !(*this == other);
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(j,i) = at(i,j);
        }
    }
    return result;
}

template <typename T>
T Matrix<T>::trace() const {
    if (!isSquare()) {
        throw std::logic_error("Trace is only defined for square matrices");
    }
    
    T result = 0;
    for (size_t i = 0; i < rows; ++i) {
        result += at(i,i);
    }
    return result;
}

template <typename T>
T Matrix<T>::determinant() const {
    if (!isSquare()) {
        throw std::logic_error("Determinant is only defined for square matrices");
    }
    
    if (rows == 1) {
        return at(0,0);
    }
    
    if (rows == 2) {
        return at(0,0) * at(1,1) - at(0,1) * at(1,0);
    }
    // Bareiss algorithm (fraction-free Gaussian elimination)
    // Works well for integer matrices (avoids fraction growth) and is O(n^3).
    // We promote computation to long double for numeric stability and cast back.
    const size_t n = rows;
    std::vector<std::vector<long double>> a(n, std::vector<long double>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            a[i][j] = static_cast<long double>(at(i,j));

    long double prev_pivot = 1.0L;
    long double det_ld = 1.0L;
    for (size_t k = 0; k < n - 1; ++k) {
        // Partial pivoting: find non-zero pivot
        size_t pivot_row = k;
        for (size_t i = k; i < n; ++i) {
            if (std::fabsl(a[i][k]) > std::fabsl(a[pivot_row][k])) {
                pivot_row = i;
            }
        }
        if (std::fabsl(a[pivot_row][k]) < 1e-18L) {
            return T(0);
        }
        if (pivot_row != k) {
            std::swap(a[pivot_row], a[k]);
            det_ld = -det_ld; // row swap changes sign
        }

        for (size_t i = k + 1; i < n; ++i) {
            for (size_t j = k + 1; j < n; ++j) {
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) / prev_pivot;
            }
            a[i][k] = 0.0L;
        }
        prev_pivot = a[k][k];
        if (std::fabsl(prev_pivot) < 1e-18L) {
            return T(0);
        }
    }
    det_ld *= a[n - 1][n - 1];
    // Cast back to T
    return static_cast<T>(det_ld);
}

template <typename T>
T Matrix<T>::determinant_naive() const {
    if (!isSquare()) {
        throw std::logic_error("Determinant is only defined for square matrices");
    }
    if (rows == 1) {
        return data[0][0];
    }
    if (rows == 2) {
        return data[0][0] * data[1][1] - data[0][1] * data[1][0];
    }
    T det = 0;
    for (size_t j = 0; j < cols; ++j) {
        Matrix submatrix(rows - 1, cols - 1);
        for (size_t i = 1; i < rows; ++i) {
            for (size_t k = 0; k < cols; ++k) {
                if (k < j) {
                    submatrix.at(i - 1, k) = at(i, k);
                } else if (k > j) {
                    submatrix.at(i - 1, k - 1) = at(i, k);
                }
            }
        }
        T sign = (j % 2 == 0) ? 1 : -1;
        det += sign * at(0, j) * submatrix.determinant_naive();
    }
    return det;
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const {
    if (!isSquare()) {
        throw std::logic_error("Inverse is only defined for square matrices");
    }

    // LUP decomposition
    Matrix L(rows, cols, 0), U(rows, cols, 0);
    std::vector<size_t> piv(rows);
    T parity = 1;
    if (!lup_decompose(L, U, piv, parity)) {
        throw std::logic_error("Matrix is singular and not invertible");
    }

    // Compute inverse by solving A x_i = e_i for each basis vector e_i
    Matrix inv(rows, cols, 0);
    for (size_t col = 0; col < cols; ++col) {
        std::vector<T> e(rows, 0);
        e[col] = 1;

        // Permute RHS according to piv: Pb
        std::vector<T> Pb(rows);
        for (size_t i = 0; i < rows; ++i) Pb[i] = e[piv[i]];

        // Forward substitution Ly = Pb (L unit lower)
        std::vector<T> y(rows, 0);
        for (size_t i = 0; i < rows; ++i) {
            T sum = Pb[i];
            for (size_t j = 0; j < i; ++j) sum -= L.at(i,j) * y[j];
            y[i] = sum; // L has 1s on diagonal
        }

        // Backward substitution Ux = y
        std::vector<T> x(rows, 0);
        for (int i = int(rows) - 1; i >= 0; --i) {
            T sum = y[size_t(i)];
            for (size_t j = size_t(i) + 1; j < cols; ++j) sum -= U.at(size_t(i), j) * x[j];
            T diag = U.at(size_t(i), size_t(i));
            if (diag == 0) throw std::logic_error("Singular matrix in back substitution");
            x[size_t(i)] = sum / diag;
        }

        // Set column of inverse
        for (size_t i = 0; i < rows; ++i) inv.at(i, col) = x[i];
    }

    return inv;
}

template <typename T>
std::vector<T> Matrix<T>::solve(const std::vector<T>& b) const {
    if (rows != b.size()) {
        throw std::invalid_argument("Dimension mismatch in solve: b size must equal number of rows");
    }
    Matrix L(rows, cols, 0), U(rows, cols, 0);
    std::vector<size_t> piv(rows);
    T parity = 1;
    if (!lup_decompose(L, U, piv, parity)) {
        throw std::logic_error("Matrix is singular");
    }

    // Permute RHS: Pb
    std::vector<T> Pb(rows);
    for (size_t i = 0; i < rows; ++i) Pb[i] = b[piv[i]];

    // Forward substitution Ly = Pb
    std::vector<T> y(rows, 0);
    for (size_t i = 0; i < rows; ++i) {
        T sum = Pb[i];
        for (size_t j = 0; j < i; ++j) sum -= L.at(i,j) * y[j];
        y[i] = sum;
    }

    // Back substitution Ux = y
    std::vector<T> x(rows, 0);
    for (int i = int(rows) - 1; i >= 0; --i) {
        T sum = y[size_t(i)];
        for (size_t j = size_t(i) + 1; j < cols; ++j) sum -= U.at(size_t(i), j) * x[j];
        T diag = U.at(size_t(i), size_t(i));
        if (diag == 0) throw std::logic_error("Singular matrix in back substitution");
        x[size_t(i)] = sum / diag;
    }
    return x;
}

template <typename T>
bool Matrix<T>::lup_decompose(Matrix& L, Matrix& U, std::vector<size_t>& piv, T& parity) const {
    if (!isSquare()) return false;
    const size_t n = rows;
    // Initialize U as a copy of *this, L as identity, piv as 0..n-1
    U = *this;
    L = Matrix::identity(n);
    piv.resize(n);
    std::iota(piv.begin(), piv.end(), 0);
    parity = 1;

    for (size_t k = 0; k < n; ++k) {
        // Pivot selection
        size_t pivot = k;
        T maxval = std::abs(U.at(k,k));
        for (size_t i = k + 1; i < n; ++i) {
            T v = std::abs(U.at(i,k));
            if (v > maxval) { maxval = v; pivot = i; }
        }
        if (maxval == T(0)) return false; // singular

        if (pivot != k) {
            // Swap rows in U
            for (size_t j = 0; j < n; ++j) std::swap(U.at(k,j), U.at(pivot,j));
            // Swap rows in L (columns < k)
            for (size_t j = 0; j < k; ++j) std::swap(L.at(k,j), L.at(pivot,j));
            std::swap(piv[k], piv[pivot]);
            parity = -parity;
        }

        // Eliminate below pivot
        for (size_t i = k + 1; i < n; ++i) {
            T factor = U.at(i,k) / U.at(k,k);
            L.at(i,k) = factor;
            U.at(i,k) = 0;
            for (size_t j = k + 1; j < n; ++j) {
                U.at(i,j) -= factor * U.at(k,j);
            }
        }
    }
    return true;
}

template <typename T>
Matrix<T> Matrix<T>::identity(size_t size) {
    Matrix result(size, size, 0);
    for (size_t i = 0; i < size; ++i) {
        result.at(i,i) = 1;
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0);
}

template <typename T>
Matrix<T> Matrix<T>::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1);
}

// Non-member operator overloads
template <typename T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& matrix) {
    return matrix * scalar;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            os << matrix.at(i,j);
            if (j < matrix.cols - 1) {
                os << " ";
            }
        }
        if (i < matrix.rows - 1) {
            os << "\n";
        }
    }
    return os;
}

template <typename T>
std::istream& operator>>(std::istream& is, Matrix<T>& matrix) {
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            is >> const_cast<Matrix<T>&>(matrix).at(i,j);
        }
    }
    return is;
}

// Explicit template instantiation for common types
template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;

// Explicit instantiation of non-member functions
template Matrix<int> operator*<int>(const int&, const Matrix<int>&);
template Matrix<float> operator*<float>(const float&, const Matrix<float>&);
template Matrix<double> operator*<double>(const double&, const Matrix<double>&);

template std::ostream& operator<< <int>(std::ostream&, const Matrix<int>&);
template std::ostream& operator<< <float>(std::ostream&, const Matrix<float>&);
template std::ostream& operator<< <double>(std::ostream&, const Matrix<double>&);

template std::istream& operator>> <int>(std::istream&, Matrix<int>&);
template std::istream& operator>> <float>(std::istream&, Matrix<float>&);
template std::istream& operator>> <double>(std::istream&, Matrix<double>&);

} // namespace matrixlib
