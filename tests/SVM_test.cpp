#include <gtest/gtest.h>
#include <SVM.h>
#include <vector>

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

Problem SAMPLE_PROBLEM{X, Y, 1, 0.001, 6, 2};
TEST(SVM_KERNELS, linear){
    std::vector<double> vector1 = {2, 3, 4};
    std::vector<double> vector2 = {2, 3, 4};
    double output = dot(vector1, vector2);
    ASSERT_DOUBLE_EQ(output, 29);
}
TEST(SVM_FUNCTIONS, predict_label) {
Kernel linear_kernel = Kernel(dot, KernelType::Linear);
SVM Estimator = SVM(SAMPLE_PROBLEM, linear_kernel);
Estimator.SMO();
for (size_t idx = 0; idx < T.size(); idx ++){
    ASSERT_EQ(Estimator.predict_label(T[idx]), TRUE_RESULT[idx]);
    }
}