#include <gtest/gtest.h>
#include "matrixlib.h"

using namespace matrixlib;

TEST(MatrixTest, DefaultConstructor) {
    Matrixi m;
    EXPECT_EQ(m.getRows(), 0);
    EXPECT_EQ(m.getCols(), 0);
    EXPECT_TRUE(m.isEmpty());
}

TEST(MatrixTest, ConstructorWithDimensions) {
    Matrixi m(2, 3, 5);
    EXPECT_EQ(m.getRows(), 2);
    EXPECT_EQ(m.getCols(), 3);
    
    for (size_t i = 0; i < m.getRows(); ++i) {
        for (size_t j = 0; j < m.getCols(); ++j) {
            EXPECT_EQ(m[i][j], 5);
        }
    }
}

TEST(MatrixTest, InitializerListConstructor) {
    Matrixi m = {{1, 2}, {3, 4}, {5, 6}};
    
    EXPECT_EQ(m.getRows(), 3);
    EXPECT_EQ(m.getCols(), 2);
    
    EXPECT_EQ(m[0][0], 1); EXPECT_EQ(m[0][1], 2);
    EXPECT_EQ(m[1][0], 3); EXPECT_EQ(m[1][1], 4);
    EXPECT_EQ(m[2][0], 5); EXPECT_EQ(m[2][1], 6);
}

TEST(MatrixTest, CopyConstructor) {
    Matrixi m1 = {{1, 2}, {3, 4}};
    Matrixi m2 = m1;  // Copy constructor
    
    EXPECT_EQ(m1.getRows(), m2.getRows());
    EXPECT_EQ(m1.getCols(), m2.getCols());
    
    for (size_t i = 0; i < m1.getRows(); ++i) {
        for (size_t j = 0; j < m1.getCols(); ++j) {
            EXPECT_EQ(m1[i][j], m2[i][j]);
        }
    }
}

TEST(MatrixTest, MoveConstructor) {
    Matrixi m1 = {{1, 2}, {3, 4}};
    Matrixi m2 = std::move(m1);  // Move constructor
    
    EXPECT_EQ(m2.getRows(), 2);
    EXPECT_EQ(m2.getCols(), 2);
    EXPECT_EQ(m2[0][0], 1);
    
    // m1 should be in a valid but unspecified state
    EXPECT_EQ(m1.getRows(), 0);
    EXPECT_EQ(m1.getCols(), 0);
}

TEST(MatrixTest, AssignmentOperator) {
    Matrixi m1 = {{1, 2}, {3, 4}};
    Matrixi m2;
    m2 = m1;  // Assignment operator
    
    EXPECT_EQ(m1.getRows(), m2.getRows());
    EXPECT_EQ(m1.getCols(), m2.getCols());
    
    for (size_t i = 0; i < m1.getRows(); ++i) {
        for (size_t j = 0; j < m1.getCols(); ++j) {
            EXPECT_EQ(m1[i][j], m2[i][j]);
        }
    }
}

TEST(MatrixTest, MoveAssignmentOperator) {
    Matrixi m1 = {{1, 2}, {3, 4}};
    Matrixi m2;
    m2 = std::move(m1);  // Move assignment operator
    
    EXPECT_EQ(m2.getRows(), 2);
    EXPECT_EQ(m2.getCols(), 2);
    EXPECT_EQ(m2[0][0], 1);
    
    // m1 should be in a valid but unspecified state
    EXPECT_EQ(m1.getRows(), 0);
    EXPECT_EQ(m1.getCols(), 0);
}

TEST(MatrixTest, Addition) {
    Matrixi m1 = {{1, 2}, {3, 4}};
    Matrixi m2 = {{5, 6}, {7, 8}};
    Matrixi expected = {{6, 8}, {10, 12}};
    
    auto result = m1 + m2;
    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, Subtraction) {
    Matrixi m1 = {{5, 6}, {7, 8}};
    Matrixi m2 = {{1, 2}, {3, 4}};
    Matrixi expected = {{4, 4}, {4, 4}};
    
    auto result = m1 - m2;
    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, MatrixMultiplication) {
    Matrixi m1 = {{1, 2, 3}, {4, 5, 6}};
    Matrixi m2 = {{7, 8}, {9, 10}, {11, 12}};
    Matrixi expected = {{58, 64}, {139, 154}};
    
    auto result = m1 * m2;
    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, ScalarMultiplication) {
    Matrixi m = {{1, 2}, {3, 4}};
    Matrixi expected1 = {{2, 4}, {6, 8}};
    Matrixi expected2 = {{3, 6}, {9, 12}};
    
    auto result1 = m * 2;
    auto result2 = 3 * m;
    
    EXPECT_EQ(result1, expected1);
    EXPECT_EQ(result2, expected2);
}

TEST(MatrixTest, Transpose) {
    Matrixi m = {{1, 2, 3}, {4, 5, 6}};
    Matrixi expected = {{1, 4}, {2, 5}, {3, 6}};
    
    auto result = m.transpose();
    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, Trace) {
    Matrixi m = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int expected = 1 + 5 + 9;
    
    auto result = m.trace();
    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, Determinant2x2) {
    Matrixi m = {{4, 6}, {3, 8}};
    int expected = 4*8 - 6*3;  // 32 - 18 = 14
    
    auto result = m.determinant();
    EXPECT_EQ(result, expected);
}

TEST(MatrixTest, IdentityMatrix) {
    auto id = Matrixi::identity(3);
    
    EXPECT_EQ(id.getRows(), 3);
    EXPECT_EQ(id.getCols(), 3);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_EQ(id[i][j], 1);
            } else {
                EXPECT_EQ(id[i][j], 0);
            }
        }
    }
}

TEST(MatrixTest, ZerosMatrix) {
    auto zeros = Matrixi::zeros(2, 3);
    
    EXPECT_EQ(zeros.getRows(), 2);
    EXPECT_EQ(zeros.getCols(), 3);
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(zeros[i][j], 0);
        }
    }
}

TEST(MatrixTest, OnesMatrix) {
    auto ones = Matrixi::ones(3, 2);
    
    EXPECT_EQ(ones.getRows(), 3);
    EXPECT_EQ(ones.getCols(), 2);
    
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(ones[i][j], 1);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
