# cython: language_level=3
from libc.stddef cimport size_t
from SVM import SVC, SVR, LinearKernel, RBFKernel, PolynomialKernel
cdef class PyKernel:
    """
    Unified Python wrapper for C++ Kernel implementations.
    
    Parameters:
        name (str): Kernel type - "linear", "rbf", or "polynomial" (or "poly").
        g (double): Gamma parameter for RBF kernel (default: 0.5).
        d (double): Degree parameter for Polynomial kernel (default: 3.0).
        b (double): Bias/offset parameter for Polynomial kernel (default: 0.0).
    """
    cdef Kernel* ptr
    cdef readonly str kernel_type

    def __cinit__(self, str name="linear", double g=0.5, double d=3.0, double b=0.0):
        self.ptr = NULL
        self.kernel_type = name.lower()

        if self.kernel_type == "linear":
            self.ptr = new LinearKernel()
        elif self.kernel_type == "rbf":
            self.ptr = new RBFKernel(g)
        elif self.kernel_type in ("polynomial", "poly"):
            self.ptr = new PolynomialKernel(d, b)
        else:
            raise ValueError(f"Unknown kernel type '{name}'. Options: 'linear', 'rbf', 'polynomial'.")

    def __dealloc__(self):
        if self.ptr != NULL:
            # virtual destructor will be called automatically
            del self.ptr
            self.ptr = NULL

    cdef Kernel* get_ptr(self):
        return self.ptr