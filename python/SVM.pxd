# SVM.pxd
from libc.stddef cimport size_t

cdef extern from "Kernel.h":
    cdef cppclass Kernel:
        double operator()(const double* vector1, const double* vector2, size_t size) except +

    cdef cppclass LinearKernel(Kernel):
        LinearKernel() except +

    cdef cppclass RBFKernel(Kernel):
        RBFKernel(double g) except +

    cdef cppclass PolynomialKernel(Kernel):
        PolynomialKernel(double d, double b) except +

cdef extern from "SVC.h" namespace "SVC":
    cdef struct Problem:
        const double* training_input
        const int* labels
        const double C
        const double tolerance
        const size_t size
        const size_t no_dim
        const double eps
        const int max_iter
        Problem(const double* training_input, const int* labels, const double C, 
                const double tolerance, const size_t size, const size_t no_dim, 
                const double eps, const int max_iter)

    cdef cppclass SVM:
        SVM(Problem prob, Kernel* k, unsigned int seed) except +
        void SMO() except +
        double predict(const double* input) except +
        int predict_label(const double* input) except +

cdef extern from "SVR.h" namespace "SVR":
    cdef struct Problem:
        const double* training_input
        const double* targets
        const double C
        const double tolerance
        const double epsilon
        const size_t size
        const size_t no_dim
        const double mu
        const int max_iter
        Problem(const double* training_input, const double* targets, const double C, 
                const double tolerance, const double epsilon, const size_t size, const size_t no_dim, 
                const double mu, const int max_iter)

    cdef cppclass SVM:
        SVM(Problem prob, Kernel* k) except +
        void SMO() except +
        double predict(const double* input) except +
