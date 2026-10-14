import numpy as np
import svm_cpp as SVM
# Testing SVC Python bindings with linear kernel
arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)

# Create a PyKernel instance
kernel = SVM.PyKernel("linear", 0.5, 1.0, 2.0)
model = SVM.PySVC()

X = np.array([[-2.0, -1.0], [-1.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
Y = np.array([-1, -1, -1, 1, 1, 1], dtype=np.int32)

T = np.array([[-1.0, -1.0], [2.0, 2.0], [3.0, 2.0]], dtype=np.float64)
TRUE_RESULT = np.array([-1, 1, 1], dtype=np.int32)
model.fit(X, Y, kernel)

result = model.predict_label(T)
print("Predictions:", result)
print("Expected:   ", TRUE_RESULT)
print("Close match:", np.array_equal(result, TRUE_RESULT))
#---------------------------------
#Testing SVR with polynomial kernel
X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
Y = np.array([3.0, 0.0, -1.0, 0.0, 3.0], dtype=np.float64)  # y = x^2 - 1

# 2. Test points: {1.5} -> expected 1.25, {-0.5} -> expected -0.75
T = np.array([[1.5], [-0.5]], dtype=np.float64)
TRUE_RESULT = np.array([1.25, -0.75], dtype=np.float64)

# 3. Create a PyKernel instance (adjust arguments depending on your PyKernel signature)
# Assuming a polynomial kernel signature like PyKernel("poly", degree, gamma, coef0)
kernel = SVM.PyKernel("poly", 1.0, 2.0, 0.0)

# 4. Instantiate and train the SVR model
model = SVM.PySVR()

# Fit parameters matching your C++ setup:
# C=100.0, tolerance=0.001, epsilon=0.01, mu=0.00001, max_iter=1000
model.fit(
    X, 
    Y, 
    kernel, 
    C=100.0, 
    tolerance=0.001, 
    epsilon=0.01, 
    mu=0.00001, 
    max_iter=1000
)

# 5. Predict on test points
predictions = model.predict(T)

print("Predictions:", predictions)
print("Expected:   ", TRUE_RESULT)
print("Close match:", np.allclose(predictions, TRUE_RESULT, atol=0.1))