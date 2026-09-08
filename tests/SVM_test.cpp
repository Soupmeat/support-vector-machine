#include <gtest/gtest.h>
#include <SVM.h>
#include <vector>

SVM Estimator = SVM({{0,2,1},{2,1,2},{2,2,1}}, {0,0,1}, 2);
TEST(SVM_FUNCTIONS, Predict) {
    ASSERT_EQ(Estimator.predict({0, 0, 0}), -1);
}