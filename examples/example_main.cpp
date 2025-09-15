#include <iostream>
#include "matrixlib.h"

int main() {
    using namespace matrixlib;
    
    std::cout << "Matrix Library Example\n";
    std::cout << "====================\n\n";

    // Create some matrices
    Matrixi a = {{1, 2, 3}, {4, 5, 6}};
    Matrixi b = {{7, 8}, {9, 10}, {11, 12}};
    
    std::cout << "Matrix A (2x3):\n" << a << "\n\n";
    std::cout << "Matrix B (3x2):\n" << b << "\n\n";
    
    // Matrix multiplication
    auto c = a * b;
    std::cout << "A * B (2x2):\n" << c << "\n\n";
    
    // Scalar multiplication
    auto d = c * 2;
    std::cout << "2 * (A * B):\n" << d << "\n\n";
    
    // Identity matrix
    auto identity = Matrixi::identity(3);
    std::cout << "Identity matrix (3x3):\n" << identity << "\n\n";
    
    // Matrix operations
    Matrixd m1 = {{1.0, 2.0}, {3.0, 4.0}};
    std::cout << "Matrix M (2x2):\n" << m1 << "\n\n";
    
    std::cout << "Determinant of M: " << m1.determinant() << "\n\n";
    
    std::cout << "Transpose of M:\n" << m1.transpose() << "\n\n";
    
    try {
        std::cout << "Inverse of M:\n" << m1.inverse() << "\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Error calculating inverse: " << e.what() << "\n";
    }
    
    return 0;
}
