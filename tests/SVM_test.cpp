#include <gtest/gtest.h>
#include <SVC.h>
#include <vector>
#include <SVR.h>

Kernel *linear_kernel = new LinearKernel();
Kernel *rbf_kernel = new RBFKernel(0.5);
Kernel *polynomial_kernel = new PolynomialKernel(2, 1);

TEST(SVM_KERNELS, operator)
{

    std::vector<double> vector1 = {2, 3, 4};
    std::vector<double> vector2 = {2, 3, 4};
    double linear_output = (*linear_kernel)(vector1, vector2);
    double rbf_output = (*rbf_kernel)(vector1, vector2);
    double polynomial_output = (*polynomial_kernel)(vector1, vector2);
    ASSERT_DOUBLE_EQ(linear_output, 29);
    ASSERT_DOUBLE_EQ(rbf_output, 1);
    ASSERT_DOUBLE_EQ(polynomial_output, 900);
}
TEST(SVC_FUNCTIONS, predict_label)
{
    // toy sample (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/tests/test_svm.py)
    std::vector<double> X = {-2.0, -1.0, -1.0, -1.0, -1.0, -2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0};
    std::vector<int> Y = {-1, -1, -1, 1, 1, 1};

    std::vector<std::vector<double>> T = {
        {-1.0, -1.0},
        {2.0, 2.0},
        {3.0, 2.0}};

    std::vector<int> TRUE_RESULT = {-1, 1, 1};

    SVC::Problem SAMPLE_PROBLEM{X, Y, 1, 0.001, 6, 2, 0.00001};
    SVC::SVM Linear_Estimator(SAMPLE_PROBLEM, linear_kernel);
    SVC::SVM RBF_Estimator(SAMPLE_PROBLEM, rbf_kernel);
    SVC::SVM Polynomial_Estimator(SAMPLE_PROBLEM, polynomial_kernel);
    Linear_Estimator.SMO();
    RBF_Estimator.SMO();
    Polynomial_Estimator.SMO();
    for (size_t idx = 0; idx < T.size(); idx++)
    {
        EXPECT_EQ(Polynomial_Estimator.predict_label(T[idx]), TRUE_RESULT[idx]);
        EXPECT_EQ(Linear_Estimator.predict_label(T[idx]), TRUE_RESULT[idx]);
        EXPECT_EQ(RBF_Estimator.predict_label(T[idx]), TRUE_RESULT[idx]);
    }
}

TEST(SVR_FUNCTIONS, predict)
{
    std::vector<std::vector<double>> X = {
        {-2.0, -1.0},
        {-1.0, -1.0},
        {-1.0, -2.0},
        {1.0, 1.0},
        {1.0, 2.0},
        {2.0, 1.0}};
    std::vector<double> Y = {-3.0, -2.0, -3.0, 2.0, 3.0, 3.0};

    std::vector<std::vector<double>> T = {
        {-1.0, -1.0},
        {2.0, 2.0},
        {3.0, 2.0}};

    std::vector<double> TRUE_RESULT = {-2.0, 4.0, 5.0};
    const double EPSILON_TOLERANCE = 1e-5;
    SVR::Problem SAMPLE_PROBLEM{X, Y, 1, 0.001, EPSILON_TOLERANCE, 6, 2, 0.00001};
    SVR::SVM Linear_Estimator(SAMPLE_PROBLEM, linear_kernel);
    Linear_Estimator.SMO();
    for (size_t idx = 0; idx < T.size(); idx++)
    {
        EXPECT_NEAR(Linear_Estimator.predict(T[idx]), TRUE_RESULT[idx], 2 * EPSILON_TOLERANCE);
    }
}

TEST(SVC_FUNCTIONS, xor_problem_rbf)
{
    std::vector<double> X = {
        1.0,  1.0,
       -1.0, -1.0,
        1.0, -1.0,
       -1.0,  1.0
    };
    std::vector<int> Y = {-1, -1, 1, 1};

    std::vector<std::vector<double>> T = {
        { 0.8,  0.8},
        {-0.8, -0.8},
        { 0.8, -0.8},
        {-0.8,  0.8}
    };
    std::vector<int> TRUE_RESULT = {-1, -1, 1, 1};

    // SVC::Problem{training_input, labels, C, tolerance, size, no_dim, eps}
    SVC::Problem PROBLEM{X, Y, 10.0, 0.001, 4, 2, 0.00001};

    SVC::SVM RBF_Estimator(PROBLEM, rbf_kernel);
    RBF_Estimator.SMO();

    for (size_t idx = 0; idx < T.size(); idx++)
    {
        EXPECT_EQ(RBF_Estimator.predict_label(T[idx]), TRUE_RESULT[idx])
            << "RBF kernel failed to classify XOR point at index " << idx;
    }
}

// 2. Concentric Circles (Radial Boundary)
// Inner ring (radius ~ 0.5) is class -1, Outer ring (radius ~ 2.0) is class +1
TEST(SVC_FUNCTIONS, concentric_circles_rbf)
{
    std::vector<double> X = {
        // Inner circle (-1)
        0.5, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, -0.5,
        // Outer circle (+1)
        2.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.0, -2.0
    };
    std::vector<int> Y = {-1, -1, -1, -1, 1, 1, 1, 1};

    std::vector<std::vector<double>> T = {
        { 0.2,  0.2}, // Deep inside inner circle
        { 1.8,  1.8}  // Deep inside outer circle
    };
    std::vector<int> TRUE_RESULT = {-1, 1};

    SVC::Problem PROBLEM{X, Y, 10.0, 0.001, 8, 2, 0.00001};
    SVC::SVM RBF_Estimator(PROBLEM, rbf_kernel);
    RBF_Estimator.SMO();

    for (size_t idx = 0; idx < T.size(); idx++)
    {
        EXPECT_EQ(RBF_Estimator.predict_label(T[idx]), TRUE_RESULT[idx]);
    }
}

// 3. Duplicate Points (Numerical Stability / Deadlock Check)
// Tests whether duplicate inputs with identical feature vectors cause infinite loops or zero-division in SMO
TEST(SVC_FUNCTIONS, duplicate_inputs_robustness)
{
    std::vector<double> X = {
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        -1.0, -1.0, -1.0, -1.0
    };
    std::vector<int> Y = {1, 1, 1, -1, -1};

    SVC::Problem PROBLEM{X, Y, 1.0, 0.001, 5, 2, 0.00001};
    SVC::SVM Estimator(PROBLEM, linear_kernel);

    // SMO must terminate cleanly without deadlock
    Estimator.SMO();

    EXPECT_EQ(Estimator.predict_label({1.0, 1.0}), 1);
    EXPECT_EQ(Estimator.predict_label({-1.0, -1.0}), -1);
}


// ============================================================================
// SVR (REGRESSION) TEST CASES
// ============================================================================

// 4. Non-Linear Curve Fitting: y = sin(x)
// Tests continuous non-linear regression using RBF kernel
TEST(SVR_FUNCTIONS, sine_wave_rbf)
{
    std::vector<std::vector<double>> X = {
        {-3.0}, {-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}, {3.0}
    };
    std::vector<double> Y;
    for (const auto& x : X) {
        Y.push_back(std::sin(x[0]));
    }

    // Unseen test mid-points
    std::vector<std::vector<double>> T = {
        {0.5},   // sin(0.5)  ~  0.4794
        {-1.57}  // sin(-1.57) ~ -0.9999
    };
    std::vector<double> TRUE_RESULT = { std::sin(0.5), std::sin(-1.57) };

    const double EPS = 0.05;
    // SVR::Problem{training_input, targets, C, tolerance, epsilon, size, no_dim, mu}
    SVR::Problem PROBLEM{X, Y, 100.0, 0.001, EPS, 7, 1, 0.00001};
    SVR::SVM RBF_Estimator(PROBLEM, rbf_kernel);
    RBF_Estimator.SMO();

    for (size_t idx = 0; idx < T.size(); idx++)
    {
        EXPECT_NEAR(RBF_Estimator.predict(T[idx]), TRUE_RESULT[idx], 3 * EPS)
            << "SVR failed on sine wave interpolation at x = " << T[idx][0];
    }
}

// 5. 3D Multivariate Linear Regression
// True model: y = 2.0*x1 - 3.0*x2 + 0.5*x3
TEST(SVR_FUNCTIONS, multivariate_linear_3d)
{
    std::vector<std::vector<double>> X = {
        { 1.0,  0.0,  0.0}, // y = 2.0
        { 0.0,  1.0,  0.0}, // y = -3.0
        { 0.0,  0.0,  1.0}, // y = 0.5
        { 1.0,  1.0,  1.0}, // y = -0.5
        {-1.0,  2.0,  0.0}  // y = -8.0
    };
    std::vector<double> Y = {2.0, -3.0, 0.5, -0.5, -8.0};

    // Test point: {2.0, 1.0, -2.0} -> y = 2(2) - 3(1) + 0.5(-2) = 0.0
    std::vector<std::vector<double>> T = {
        {2.0, 1.0, -2.0}
    };
    std::vector<double> TRUE_RESULT = {0.0};

    const double EPS = 0.001;
    SVR::Problem PROBLEM{X, Y, 500.0, 0.0001, EPS, 5, 3, 0.00001};
    SVR::SVM Linear_Estimator(PROBLEM, linear_kernel);
    Linear_Estimator.SMO();

    for (size_t idx = 0; idx < T.size(); idx++)
    {
        EXPECT_NEAR(Linear_Estimator.predict(T[idx]), TRUE_RESULT[idx], 2 * EPS);
    }
}

// 6. Quadratic Curve Fitting: y = x^2 - 1
// Tests Polynomial Kernel (degree 2, bias 1)
TEST(SVR_FUNCTIONS, quadratic_polynomial_kernel)
{
    std::vector<std::vector<double>> X = {
        {-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}
    };
    std::vector<double> Y = {3.0, 0.0, -1.0, 0.0, 3.0}; // y = x^2 - 1

    // Test point: {1.5} -> y = 1.5^2 - 1 = 1.25
    std::vector<std::vector<double>> T = {
        {1.5}, {-0.5}
    };
    std::vector<double> TRUE_RESULT = {1.25, -0.75};

    const double EPS = 0.01;
    SVR::Problem PROBLEM{X, Y, 100.0, 0.001, EPS, 5, 1, 0.00001};
    SVR::SVM Poly_Estimator(PROBLEM, polynomial_kernel);
    Poly_Estimator.SMO();

    for (size_t idx = 0; idx < T.size(); idx++)
    {
        EXPECT_NEAR(Poly_Estimator.predict(T[idx]), TRUE_RESULT[idx], 3 * EPS);
    }
}
