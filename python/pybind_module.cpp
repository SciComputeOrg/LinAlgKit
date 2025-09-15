#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "matrixlib.h"

namespace py = pybind11;
using matrixlib::Matrix;
using matrixlib::Matrixd;

// Helper for 2D indexing via tuple (i, j)
template<typename T>
T get_item(const Matrix<T>& m, std::pair<size_t, size_t> idx) {
    auto [i, j] = idx;
    if (i >= m.getRows() || j >= m.getCols()) {
        throw std::out_of_range("index out of range");
    }
    return m[i][j];
}

template<typename T>
void set_item(Matrix<T>& m, std::pair<size_t, size_t> idx, const T& value) {
    auto [i, j] = idx;
    if (i >= m.getRows() || j >= m.getCols()) {
        throw std::out_of_range("index out of range");
    }
    m[i][j] = value;
}

PYBIND11_MODULE(matrixlib_py, m) {
    m.doc() = "Python bindings for matrixlib with NumPy interop";

    // Helpers for NumPy conversion
    auto to_numpy_d = [](const Matrix<double>& a) {
        const size_t rows = a.getRows();
        const size_t cols = a.getCols();
        py::array_t<double> arr({rows, cols});
        auto buf = arr.mutable_unchecked<2>();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                buf(i, j) = a[i][j];
            }
        }
        return arr;
    };

    auto from_numpy_d = [](const py::array& array) {
        auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>(array);
        if (arr.ndim() != 2) throw std::invalid_argument("Expected 2D numpy array");
        size_t rows = static_cast<size_t>(arr.shape(0));
        size_t cols = static_cast<size_t>(arr.shape(1));
        Matrix<double> m(rows, cols, 0.0);
        auto r = arr.unchecked<2>();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                m[i][j] = r(i, j);
        return m;
    };

    auto to_numpy_f = [](const Matrix<float>& a) {
        const size_t rows = a.getRows();
        const size_t cols = a.getCols();
        py::array_t<float> arr({rows, cols});
        auto buf = arr.mutable_unchecked<2>();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                buf(i, j) = a[i][j];
        return arr;
    };

    auto from_numpy_f = [](const py::array& array) {
        auto arr = py::array_t<float, py::array::c_style | py::array::forcecast>(array);
        if (arr.ndim() != 2) throw std::invalid_argument("Expected 2D numpy array");
        size_t rows = static_cast<size_t>(arr.shape(0));
        size_t cols = static_cast<size_t>(arr.shape(1));
        Matrix<float> m(rows, cols, 0.0f);
        auto r = arr.unchecked<2>();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                m[i][j] = r(i, j);
        return m;
    };

    auto to_numpy_i = [](const Matrix<int>& a) {
        const size_t rows = a.getRows();
        const size_t cols = a.getCols();
        py::array_t<int> arr({rows, cols});
        auto buf = arr.mutable_unchecked<2>();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                buf(i, j) = a[i][j];
        return arr;
    };

    auto from_numpy_i = [](const py::array& array) {
        auto arr = py::array_t<int, py::array::c_style | py::array::forcecast>(array);
        if (arr.ndim() != 2) throw std::invalid_argument("Expected 2D numpy array");
        size_t rows = static_cast<size_t>(arr.shape(0));
        size_t cols = static_cast<size_t>(arr.shape(1));
        Matrix<int> m(rows, cols, 0);
        auto r = arr.unchecked<2>();
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                m[i][j] = r(i, j);
        return m;
    };

    py::class_<Matrix<double>>(m, "Matrix")
        .def(py::init<>())
        .def(py::init<size_t, size_t, const double&>(),
             py::arg("rows"), py::arg("cols"), py::arg("value") = 0.0)
        .def_property_readonly("rows", &Matrix<double>::getRows)
        .def_property_readonly("cols", &Matrix<double>::getCols)
        .def("getRows", &Matrix<double>::getRows)
        .def("getCols", &Matrix<double>::getCols)
        .def("transpose", &Matrix<double>::transpose)
        .def("trace", &Matrix<double>::trace)
        .def("determinant", &Matrix<double>::determinant)
        .def("inverse", &Matrix<double>::inverse)
        .def("is_square", &Matrix<double>::isSquare)
        .def("is_empty", &Matrix<double>::isEmpty)
        .def("__getitem__", &get_item<double>)
        .def("__setitem__", &set_item<double>)
        .def("__eq__", &Matrix<double>::operator==)
        .def("__ne__", &Matrix<double>::operator!=)
        .def("__add__", [](const Matrix<double>& a, const Matrix<double>& b){ return a + b; })
        .def("__sub__", [](const Matrix<double>& a, const Matrix<double>& b){ return a - b; })
        .def("__mul__", py::overload_cast<const Matrix<double>&>(&Matrix<double>::operator*, py::const_))
        .def("__rmul__", [](const Matrix<double>& a, double s){ return s * a; })
        .def("scale", py::overload_cast<const double&>(&Matrix<double>::operator*, py::const_), py::arg("scalar"))
        .def_static("identity", &Matrix<double>::identity, py::arg("size"))
        .def_static("zeros", &Matrix<double>::zeros, py::arg("rows"), py::arg("cols"))
        .def_static("ones", &Matrix<double>::ones, py::arg("rows"), py::arg("cols"))
        .def("to_numpy", to_numpy_d)
        .def_static("from_numpy", from_numpy_d, py::arg("array"));

    // Matrix<float> as MatrixF
    py::class_<Matrix<float>>(m, "MatrixF")
        .def(py::init<>())
        .def(py::init<size_t, size_t, const float&>(),
             py::arg("rows"), py::arg("cols"), py::arg("value") = 0.0f)
        .def_property_readonly("rows", &Matrix<float>::getRows)
        .def_property_readonly("cols", &Matrix<float>::getCols)
        .def("transpose", &Matrix<float>::transpose)
        .def("trace", &Matrix<float>::trace)
        .def("determinant", &Matrix<float>::determinant)
        .def("inverse", &Matrix<float>::inverse)
        .def("__add__", [](const Matrix<float>& a, const Matrix<float>& b){ return a + b; })
        .def("__sub__", [](const Matrix<float>& a, const Matrix<float>& b){ return a - b; })
        .def("__mul__", py::overload_cast<const Matrix<float>&>(&Matrix<float>::operator*, py::const_))
        .def_static("identity", &Matrix<float>::identity)
        .def_static("zeros", &Matrix<float>::zeros)
        .def_static("ones", &Matrix<float>::ones)
        .def("to_numpy", to_numpy_f)
        .def_static("from_numpy", from_numpy_f, py::arg("array"));

    // Matrix<int> as MatrixI
    py::class_<Matrix<int>>(m, "MatrixI")
        .def(py::init<>())
        .def(py::init<size_t, size_t, const int&>(),
             py::arg("rows"), py::arg("cols"), py::arg("value") = 0)
        .def_property_readonly("rows", &Matrix<int>::getRows)
        .def_property_readonly("cols", &Matrix<int>::getCols)
        .def("transpose", &Matrix<int>::transpose)
        .def("trace", &Matrix<int>::trace)
        .def("determinant", &Matrix<int>::determinant)
        .def("__add__", [](const Matrix<int>& a, const Matrix<int>& b){ return a + b; })
        .def("__sub__", [](const Matrix<int>& a, const Matrix<int>& b){ return a - b; })
        .def("__mul__", py::overload_cast<const Matrix<int>&>(&Matrix<int>::operator*, py::const_))
        .def_static("identity", &Matrix<int>::identity)
        .def_static("zeros", &Matrix<int>::zeros)
        .def_static("ones", &Matrix<int>::ones)
        .def("to_numpy", to_numpy_i)
        .def_static("from_numpy", from_numpy_i, py::arg("array"));
}
