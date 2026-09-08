#include <SVM.h>
#include <vector>
#include <iostream> 
#include <omp.h>

SVM::SVM(std::vector<std::vector<double>> training_inputt, std::vector<int> labelss, double CC):
            training_input(std::move(training_inputt)),
            labels(std::move(labelss)), 
            C(CC),
            objective(std::numeric_limits<double>::min()),
            alpha(labels.size(), 0),
            weights(training_input[0].size(), 0),
            bias(0)
            {
            }

void SVM::compute_weights() {
    for (size_t train_inst = 0; train_inst < training_input.size(); train_inst++){
        for (size_t dim = 0; dim < weights.size(); dim ++){
            weights[dim] += alpha[train_inst] * labels[train_inst] * training_input[train_inst][dim];
        }
    }
}
int SVM::predict(std::vector<double> input) {
    double output = -bias;
    for (size_t i = 0; i < input.size(); i++){
        output += weights[i] * input[i];
    }
    if (output > 0) 
        return 1;
    return -1;
}
