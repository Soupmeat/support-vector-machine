#include <gtest/gtest.h>
#include <SVM.h>
#include <vector>

Kernel* linear_kernel = new LinearKernel();
Kernel* rbf_kernel = new RBFKernel(0.5);
Kernel* polynomial_kernel = new PolynomialKernel(2, 1);
// toy sample (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/tests/test_svm.py)
std::vector<std::vector<double>> X = {
    {-2.0, -1.0}, 
    {-1.0, -1.0}, 
    {-1.0, -2.0}, 
    { 1.0,  1.0}, 
    { 1.0,  2.0}, 
    { 2.0,  1.0}
};

std::vector<int> Y = {-1, -1, -1, 1, 1, 1};

std::vector<std::vector<double>> T = {
    {-1.0, -1.0}, 
    { 2.0,  2.0}, 
    { 3.0,  2.0}
};

std::vector<int> TRUE_RESULT = {-1, 1, 1};

Problem SAMPLE_PROBLEM{X, Y, 1, 0.001, 6, 2,0.00001};
TEST(SVM_KERNELS, operator){
    std::vector<double> vector1 = {2, 3, 4};
    std::vector<double> vector2 = {2, 3, 4};
    double linear_output = (*linear_kernel)(vector1, vector2);
    double rbf_output = (*rbf_kernel)(vector1, vector2);
    double polynomial_output = (*polynomial_kernel)(vector1, vector2);
    ASSERT_DOUBLE_EQ(linear_output, 29);
    ASSERT_DOUBLE_EQ(rbf_output, 1);
    ASSERT_DOUBLE_EQ(polynomial_output, 900);
}
TEST(SVM_FUNCTIONS, predict_label) {
SVM Linear_Estimator = SVM(SAMPLE_PROBLEM, linear_kernel);
SVM RBF_Estimator = SVM(SAMPLE_PROBLEM, rbf_kernel);
SVM Polynomial_Estimator = SVM(SAMPLE_PROBLEM, polynomial_kernel);
Linear_Estimator.SMO();
RBF_Estimator.SMO();
Polynomial_Estimator.SMO();
for (size_t idx = 0; idx < T.size(); idx ++){
    EXPECT_EQ(Polynomial_Estimator.predict_label(T[idx]), TRUE_RESULT[idx]);
    EXPECT_EQ(Linear_Estimator.predict_label(T[idx]), TRUE_RESULT[idx]);
    EXPECT_EQ(RBF_Estimator.predict_label(T[idx]), TRUE_RESULT[idx]);
    }
}