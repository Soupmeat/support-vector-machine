#include "SVM.h"
#include <vector>
#include <iostream> 

SVM::SVM(std::vector<std::vector<double>> training_inputt, std::vector<int> labelss, double CC):
            training_input(std::move(training_inputt)),
            labels(std::move(labelss)), 
            C(CC),
            objective(std::numeric_limits<double>::min()),
            alpha(labels.size(), 0)
            {
            }
void SVM::step() {
}

std::vector<std::vector<double>> training_data = {{0,1,2},{91,2,3}};
std::vector<int> labels = {0,1};
double l1_regularization = 2.0;

int main() {
    SVM Estimator = SVM(training_data, labels, l1_regularization);
}