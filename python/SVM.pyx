from libc.stddef cimport size_t
import numpy as np
cimport numpy as cnp
from SVM cimport *
cdef void _validate_and_prepare_data(object X, object y, object kernel):
    """
    Helper function to validate, cast, and ensure contiguous memory for X and y 
    before passing them to the C++ SVM engine.
    """
    if X is None or y is None:
        raise ValueError("Input arrays X and y cannot be None.")


    # 2. Comprehensive Shape Validations
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for features X, got {X.ndim}D array.")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D array for targets y, got {y.ndim}D array.")
        
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Sample size mismatch: X has {X.shape[0]} samples, "
            f"but y has {y.shape[0]} target values."
        )
        
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Input feature matrix X cannot be empty.")

    # 3. Cleanliness / NaN Checks
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Input data contains NaN or infinite values.")
cdef cnp.ndarray[double, ndim=2, mode="c"] _validate_and_prepare_predict_input(object X, size_t expected_features) except *:
    """
    Validates and prepares input X for prediction:
    - Casts to float64 numpy array.
    - Accepts 1D (single sample) or 2D (batch) arrays.
    - Reshapes 1D arrays to (1, expected_features).
    - Checks for NaN or infinite values.
    - Guarantees C-contiguous row-major layout.
    """
    # 1. Convert input safely to a NumPy array
    cdef cnp.ndarray arr
    try:
        arr = np.array(X, dtype=np.float64, copy=False)
    except Exception as e:
        raise ValueError(f"Could not convert input X to a float64 array: {e}")
    
    # 2. Check dimensions and feature alignment
    if arr.ndim == 1:
        if arr.shape[0] != expected_features:
            raise ValueError(f"Feature dimension mismatch: expected {expected_features} features, got 1D array with {arr.shape[0]} elements.")
        # Promote single 1D sample vector to a 2D row vector (1, n_features)
        arr = arr.reshape(1, expected_features)
    elif arr.ndim == 2:
        if arr.shape[1] != expected_features:
            raise ValueError(f"Feature dimension mismatch: expected {expected_features} features, got {arr.shape[1]} columns.")
    else:
        raise ValueError(f"Input X must be 1D or 2D, but got {arr.ndim} dimensions.")
        
    # 3. Check for NaN or infinite values
    if not np.isfinite(arr).all():
        raise ValueError("Input X contains NaN or infinite values.")
        
    # 4. Ensure strictly C-contiguous row-major layout
    return np.ascontiguousarray(arr, dtype=np.float64)
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

    def __cinit__(self, str name="linear", double g=0.5, int d=3, double b=0.0):
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

    def fit(self, object X, 
                  object y, 
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

        _validate_and_prepare_data(X, y, kernel)
        cdef cnp.ndarray[double, ndim=2, mode="c"] X_arr = np.ascontiguousarray(X, dtype=np.float64)
        cdef cnp.ndarray[int, ndim=1, mode="c"] y_arr = np.ascontiguousarray(y, dtype=np.int32)

        self._X = X_arr
        self._y = y_arr
        self.kernel = kernel

        # 4. Construct C++ Problem struct using its C++ constructor for const members
        self.prob = new SVC_Problem(
            <double *> &X_arr[0, 0],      # pointer to raw continuous array buffer
            <int *> &y_arr[0],         # pointer to target array buffer
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


    def predict(self, object X):
        """Calculates continuous margin values for input feature vectors."""
        if self.cpp_svm == NULL:
            raise RuntimeError("Model must be fitted before calling predict.")
        cdef size_t expected_features = self._X.shape[1]
        cdef cnp.ndarray[double, ndim=2, mode="c"] X_arr = _validate_and_prepare_predict_input(X, expected_features)
        cdef size_t n_samples = X_arr.shape[0]
        cdef cnp.ndarray[double, ndim=1, mode="c"] predictions = np.zeros(n_samples, dtype=np.float64)
        

        cdef size_t i
        for i in range(n_samples):
            # Pass raw pointer to starting address of each row
            predictions[i] = self.cpp_svm.predict(&X_arr[i, 0])

        return predictions

    def predict_label(self, double[:, ::1] X):
        """Calculates predicted discrete class labels (+1 / -1) for input feature vectors."""
        if self.cpp_svm == NULL:
            raise RuntimeError("Model must be fitted before calling predict.")
        cdef size_t expected_features = self._X.shape[1]
        cdef cnp.ndarray[double, ndim=2, mode="c"] X_arr = _validate_and_prepare_predict_input(X, expected_features)
        cdef size_t n_samples = X_arr.shape[0]
        cdef cnp.ndarray[int, ndim=1, mode="c"] labels = np.zeros(n_samples, dtype=np.int32)


        cdef size_t i
        for i in range(n_samples):
            # Pass raw pointer to starting address of each row
            labels[i] = self.cpp_svm.predict_label(&X_arr[i, 0])

        return labels

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

    def fit(self, object X, 
                  object y, 
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
        
        _validate_and_prepare_data(X, y, kernel)
        cdef cnp.ndarray[double, ndim=2, mode="c"] X_arr = np.ascontiguousarray(X, dtype=np.float64)
        cdef cnp.ndarray[double, ndim=1, mode="c"] y_arr = np.ascontiguousarray(y, dtype=np.float64)

        # 3. Bind backing objects to prevent Python Garbage Collection from freeing buffers
        self._X = X_arr
        self._y = y_arr
        self.kernel = kernel

        # 4. Construct C++ Problem struct matching the 9 constructor parameters:
        # SVR_Problem(training_input, targets, C, tolerance, epsilon, size, no_dim, mu, max_iter)
        self.prob = new SVR_Problem(
            <double *> &X_arr[0, 0],      # training_input
            <double *> &y_arr[0],         # targets (continuous values for SVR)
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

    def predict(self, object X):
        """Calculates continuous predicted regression values for input feature vectors."""
        if self.cpp_svm == NULL:
            raise RuntimeError("Model must be fitted before calling predict.")
        cdef size_t expected_features = self._X.shape[1]
        cdef cnp.ndarray[double, ndim=2, mode="c"] X_arr = _validate_and_prepare_predict_input(X, expected_features)
        cdef size_t n_samples = X_arr.shape[0]
        cdef cnp.ndarray[double, ndim=1, mode="c"] predictions = np.zeros(n_samples, dtype=np.float64)

        cdef size_t i
        for i in range(n_samples):
            # Pass raw pointer to starting address of each row
            predictions[i] = self.cpp_svm.predict(&X_arr[i, 0])

        return predictions