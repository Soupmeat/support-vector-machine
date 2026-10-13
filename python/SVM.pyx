from libc.stddef cimport size_t
import numpy as np
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

cdef class PySVC:
    """Python wrapper for C++ Support Vector Classifier (SVC)."""
    cdef SVC_SVM* cpp_svm
    cdef PyKernel kernel
    cdef object _X  # Keeps underlying data container alive in memory
    cdef object _y  # Keeps label container alive in memory
    cdef SVC_Problem* prob

    def __cinit__(self):
        self.cpp_svm = NULL
        self.prob = NULL

    def __dealloc__(self):
        if self.cpp_svm != NULL:
            del self.cpp_svm
            self.cpp_svm = NULL
        if self.prob != NULL:
            del self.prob
            self.prob = NULL

    def fit(self, double[:, ::1] X, 
                  int[::1] y, 
                  PyKernel kernel, 
                  double C=1.0, 
                  double tolerance=1e-3, 
                  double eps=1e-5, 
                  int max_iter=1000, 
                  unsigned int seed=42):
        """
        Fits the C++ SVC model.
        
        Dimensions (`size` and `no_dim`) are read directly from the C-contiguous 
        memoryview shape (`X.shape[0]` and `X.shape[1]`).
        """
        # 1. Read size and dimensions directly from the memoryview shape
        cdef size_t n_samples = X.shape[0]
        cdef size_t n_features = X.shape[1]

        # 2. Prevent memory leaks across re-fits
        if self.cpp_svm != NULL:
            del self.cpp_svm
            self.cpp_svm = NULL

        if self.prob != NULL:
            del self.prob
            self.prob = NULL

        # 3. Bind backing objects to prevent Python Garbage Collection from freeing buffers
        self._X = X.base if X.base is not None else X
        self._y = y.base if y.base is not None else y
        self.kernel = kernel

        # 4. Construct C++ Problem struct using its C++ constructor for const members
        self.prob = new SVC_Problem(
            <double *> &X[0, 0],      # pointer to raw continuous array buffer
            <int *> &y[0],         # pointer to target array buffer
            <double> C, 
            <double> tolerance, 
            <size_t> n_samples, 
            <size_t> n_features, 
            <double> eps, 
            <int> max_iter
        )

        # 5. Allocate and train C++ SVM classifier on the heap
        self.cpp_svm = new SVC_SVM(self.prob[0], self.kernel.get_ptr(), seed)
        self.cpp_svm.SMO()


    def predict(self, double[:, ::1] X):
        """Calculates continuous margin values for input feature vectors."""
        cdef size_t n_samples = X.shape[0]
        
        cdef double[::1] predictions = np.zeros(n_samples, dtype=np.float64)

        cdef size_t i
        for i in range(n_samples):
            # Pass raw pointer to starting address of each row
            predictions[i] = self.cpp_svm.predict(&X[i, 0])

        return predictions.base

    def predict_label(self, double[:, ::1] X):
        """Calculates predicted discrete class labels (+1 / -1) for input feature vectors."""
        cdef size_t n_samples = X.shape[0]

        import numpy as np
        cdef int[::1] labels = np.zeros(n_samples, dtype=np.int32)

        cdef size_t i
        for i in range(n_samples):
            # Pass raw pointer to starting address of each row
            labels[i] = self.cpp_svm.predict_label(&X[i, 0])

        return labels.base

cdef class PySVR:
    """Python wrapper for C++ Support Vector Regressor (SVR)."""
    cdef SVR_SVM* cpp_svm
    cdef PyKernel kernel
    cdef object _X  # Keeps underlying data container alive in memory
    cdef object _y  # Keeps target container alive in memory
    cdef SVR_Problem* prob

    def __cinit__(self):
        self.cpp_svm = NULL
        self.prob = NULL

    def __dealloc__(self):
        if self.cpp_svm != NULL:
            del self.cpp_svm
            self.cpp_svm = NULL
        if self.prob != NULL:
            del self.prob
            self.prob = NULL

    def fit(self, double[:, ::1] X, 
                  double[::1] y, 
                  PyKernel kernel, 
                  double C=1.0, 
                  double tolerance=1e-3, 
                  double epsilon=0.1, 
                  double mu=0.1, 
                  int max_iter=1000):
        """
        Fits the C++ SVR model.
        
        Dimensions (`size` and `no_dim`) are read directly from the C-contiguous 
        memoryview shape (`X.shape[0]` and `X.shape[1]`).
        """
        # 1. Read size and dimensions directly from the memoryview shape
        cdef size_t n_samples = X.shape[0]
        cdef size_t n_features = X.shape[1]

        # 2. Prevent memory leaks across re-fits
        if self.cpp_svm != NULL:
            del self.cpp_svm
            self.cpp_svm = NULL

        if self.prob != NULL:
            del self.prob
            self.prob = NULL

        # 3. Bind backing objects to prevent Python Garbage Collection from freeing buffers
        self._X = X.base if X.base is not None else X
        self._y = y.base if y.base is not None else y
        self.kernel = kernel

        # 4. Construct C++ Problem struct matching the 9 constructor parameters:
        # SVR_Problem(training_input, targets, C, tolerance, epsilon, size, no_dim, mu, max_iter)
        self.prob = new SVR_Problem(
            <double *> &X[0, 0],      # training_input
            <double *> &y[0],         # targets (continuous values for SVR)
            <double> C, 
            <double> tolerance, 
            <double> epsilon, 
            <size_t> n_samples,       # size
            <size_t> n_features,      # no_dim
            <double> mu, 
            <int> max_iter
        )

        # 5. Allocate and train C++ SVM regressor on the heap
        self.cpp_svm = new SVR_SVM(self.prob[0], self.kernel.get_ptr())
        self.cpp_svm.SMO()

    def predict(self, double[:, ::1] X):
        """Calculates continuous predicted regression values for input feature vectors."""
        cdef size_t n_samples = X.shape[0]
        
        cdef double[::1] predictions = np.zeros(n_samples, dtype=np.float64)

        cdef size_t i
        for i in range(n_samples):
            # Pass raw pointer to starting address of each row
            predictions[i] = self.cpp_svm.predict(&X[i, 0])

        return predictions.base