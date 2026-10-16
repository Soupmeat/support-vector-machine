import pytest
import numpy as np
import svm_cpp
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

@pytest.fixture
def binary_iris_data():
    """Load, scale [0,1], and prepare a binary classification subset of the Iris dataset."""
    iris = load_iris()
    X = MinMaxScaler().fit_transform(iris.data)
    y = iris.target
    
    mask = y < 2
    X = X[mask]
    y = y[mask]
    
    y_binary = np.where(y == 1, 1, -1)
    return X, y_binary

@pytest.fixture
def classification_data():
    """Generates and scales [0,1] a synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, n_redundant=1, n_classes=2, class_sep=2.0, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_binary = np.where(y_train == 1, 1, -1)
    y_test_binary = np.where(y_test == 1, 1, -1)
    
    return X_train_scaled, X_test_scaled, y_train_binary, y_test_binary

@pytest.fixture
def regression_data():
    """Generates and scales [0,1] a clean synthetic regression dataset for SVR evaluation."""
    np.random.seed(42)
    X = np.random.rand(150, 3)  # Uniform [0, 1) to ensure non-negative dot products
    y = (X[:, 0] * 2.5 + X[:, 1] * 1.2 + 0.5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


# ==========================================
# 1. Kernel Configuration & Error Tests
# ==========================================

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

def test_unsupported_kernel_raises():
    """Test that invalid kernel string configurations throw a ValueError."""
    with pytest.raises(ValueError):
        svm_cpp.PyKernel("nonexistent_kernel_type")


# ==========================================
# 2. SVC Integration & Kernel Matrix Tests
# ==========================================

def test_svc_iris_binary_classification(binary_iris_data):
    """Test SVC on the scaled binary Iris dataset."""
    X, y_binary = binary_iris_data

    model = svm_cpp.PySVC()
    kernel = svm_cpp.PyKernel("linear")
    
    model.fit(
        X, y_binary, kernel, 
        C=1.0, tolerance=1e-3, eps=1e-5, max_iter=2000, seed=42
    )

    preds = model.predict_label(X)
    assert preds.shape == X.shape[:1]
    assert preds.dtype == np.int32
    
    accuracy = np.mean(preds == y_binary)
    assert accuracy >= 0.95, f"Expected near-perfect accuracy on binary iris data, got {accuracy}"

@pytest.mark.parametrize("kernel_name,kernel_obj", [
    ("linear", svm_cpp.PyKernel("linear")),
    ("rbf", svm_cpp.PyKernel("rbf", g=0.5)),
    ("poly", svm_cpp.PyKernel("poly", d=2.0, b=1.0)),
    ("polynomial", svm_cpp.PyKernel("polynomial", d=2.0, b=1.0))
])
def test_svc_all_kernels_parametrized(classification_data, kernel_name, kernel_obj):
    """Parameterized test ensuring all scaled SVC kernels achieve high accuracy (>80%)."""
    X_train, X_test, y_train, y_test = classification_data

    model = svm_cpp.PySVC()
    model.fit(
        X_train, y_train, kernel_obj, 
        C=1.0, tolerance=1e-3, eps=1e-5, max_iter=3000, seed=42
    )

    preds = model.predict_label(X_test)
    assert preds.shape == y_test.shape
    assert preds.dtype == np.int32

    accuracy = np.mean(preds == y_test)
    assert accuracy > 0.80, f"SVC with kernel '{kernel_name}' underperformed on normalized data (accuracy: {accuracy:.2f})"


# ==========================================
# 3. SVR Regression, R^2, & Kernel Matrix Tests
# ==========================================

@pytest.mark.parametrize("kernel_name,kernel_obj", [
    ("linear", svm_cpp.PyKernel("linear")),
    ("rbf", svm_cpp.PyKernel("rbf", g=0.1)),
    ("poly", svm_cpp.PyKernel("poly", d=2.0, b=1.0)),
    ("polynomial", svm_cpp.PyKernel("polynomial", d=2.0, b=0.0))
])
def test_svr_all_kernels_parametrized(regression_data, kernel_name, kernel_obj):
    """Parameterized sanity test ensuring all SVR kernels fit, predict, and avoid flatlining/NaNs."""
    X_train, X_test, y_train, y_test = regression_data

    model = svm_cpp.PySVR()
    model.fit(
        X_train, y_train, kernel_obj, 
        C=10.0, tolerance=1e-3, epsilon=0.1, mu=0.1, max_iter=3000
    )

    preds = model.predict(X_test)
    
    assert preds.shape == y_test.shape
    assert preds.dtype == np.float64
    assert not np.isnan(preds).any()
    assert np.std(preds) > 0, f"SVR with kernel '{kernel_name}' collapsed to a constant value."
    
    r2 = r2_score(y_test, preds)
    assert r2 > 0.75, f"SVR with kernel '{kernel_name}' underperformed (R^2: {r2:.2f})"


def test_svr_regression_r2_performance(regression_data):
    """Dedicated regression performance test ensuring SVR achieves high R^2 fit quality."""
    X_train, X_test, y_train, y_test = regression_data

    model = svm_cpp.PySVR()
    kernel = svm_cpp.PyKernel("linear")
    
    model.fit(
        X_train, y_train, kernel, 
        C=10.0, tolerance=1e-3, epsilon=0.01, mu=0.1, max_iter=3000
    )

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    assert r2 > 0.75, f"Expected strong SVR R^2 score > 0.75, got {r2:.2f}"
    assert rmse < 0.8, f"Expected low RMSE, got {rmse:.2f}"