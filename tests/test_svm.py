import pytest
import numpy as np
import svm_cpp
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, mean_squared_error

def test_kernel_configurations():
    """Test that all kernels instantiate properly with valid parameters."""
    kernels = [
        ("linear", svm_cpp.PyKernel("linear")),
        ("rbf", svm_cpp.PyKernel("rbf", g=0.5)),
        ("poly", svm_cpp.PyKernel("poly", d=3.0, b=1.0)),
        ("polynomial", svm_cpp.PyKernel("polynomial", d=2.0, b=0.0)),
    ]
    
    for name, kernel in kernels:
        assert kernel is not None
        assert kernel.kernel_type in ["linear", "rbf", "poly", "polynomial"]

def test_svc_iris_binary_classification():
    """Test SVC on the classic scikit-learn Iris dataset (binary subset), 
    mimicking scikit-learn's integration test patterns."""
    # Load standard iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Convert to a binary classification problem (classes 0 and 1)
    mask = y < 2
    X = X[mask]
    y = y[mask]

    # Map labels from {0, 1} to your C++ engine's expected {-1, +1} convention
    y_binary = np.where(y == 1, 1, -1).astype(np.int32)

    model = svm_cpp.PySVC()
    kernel = svm_cpp.PyKernel("linear")
    
    # Fit the model using your exact signature
    model.fit(
        X, y_binary, kernel, 
        C=1.0, 
        tolerance=1e-3, 
        eps=1e-5, 
        max_iter=2000, 
        seed=42
    )

    # Predict labels
    preds = model.predict_label(X)
    
    # Assertions on shape and types
    assert preds.shape == X.shape[:1]
    assert preds.dtype == np.int32
    
    # Iris binary split is exceptionally clean; accuracy should be 100% or close to it
    accuracy = np.mean(preds == y_binary)
    assert accuracy >= 0.95, f"Expected near-perfect accuracy on binary iris data, got {accuracy}"

def test_svc_binary_classification():
    """Test SVC on a simple linearly separable binary classification problem, 
    matching your exact fit() signature with the kernel passed as an argument."""
    np.random.seed(42)
    
    # Generate clean mock data (50 samples, 2 features)
    X = np.random.randn(50, 2)
    # Binary classification labels: +1 or -1 (int32)
    y = np.where((X[:, 0] + X[:, 1]) > 0, 1, -1).astype(np.int32)

    model = svm_cpp.PySVC()
    kernel = svm_cpp.PyKernel("linear")
    
    # Fit the model passing the kernel directly to fit()
    model.fit(
        X, y, kernel, 
        C=1.0, 
        tolerance=1e-3, 
        eps=1e-5, 
        max_iter=1000, 
        seed=42
    )

    # Test prediction labels (+1 / -1)
    preds = model.predict_label(X)
    assert preds.shape == (50,)
    assert preds.dtype == np.int32
    
    # Check continuous margin outputs via predict()
    margins = model.predict(X)
    assert margins.shape == (50,)
    assert margins.dtype == np.float64

    # High accuracy check on clean linear split
    accuracy = np.mean(preds == y)
    assert accuracy > 0.85, f"Expected high accuracy on separable data, got {accuracy}"

def test_svr_regression():
    """Test SVR on a 1D/2D regression problem, matching your SVR fit signature."""
    np.random.seed(42)
    
    # Generate mock continuous regression targets (float64)
    X = np.random.randn(40, 2)
    y = (X[:, 0] * 1.5 + 2.0).astype(np.float64)

    model = svm_cpp.PySVR()
    kernel = svm_cpp.PyKernel("linear")
    
    # Fit SVR passing the kernel directly
    model.fit(
        X, y, kernel, 
        C=1.0, 
        tolerance=1e-3, 
        epsilon=0.1, 
        mu=0.1, 
        max_iter=1000
    )

    # Predict continuous values
    preds = model.predict(X)
    # Assertions
    assert preds.shape == (40,)
    assert preds.dtype == np.float64
    assert not np.isnan(preds).any()
    assert np.std(preds) > 0
    # --- Professional Regression Metrics ---
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    # Assert that the model captures strong variance and has low error
    assert r2 > 0.85, f"Expected R^2 score > 0.85, got {r2}"
    assert rmse < 0.3, f"Expected low RMSE, got {rmse}"

def test_unsupported_kernel_raises():
    """Test that invalid kernel string configurations throw a ValueError."""
    with pytest.raises(ValueError):
        svm_cpp.PyKernel("nonexistent_kernel_type")