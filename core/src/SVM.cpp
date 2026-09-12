#include <SVM.h>
#include <vector>

double dot(std::vector<double> vector1, std::vector<double> vector2){
    size_t size = vector1.size();
    double answer = 0;
    for (size_t idx = 0; idx < size; idx++){
        answer += vector1[idx] * vector2[idx];
    }
    return answer;
}
SVM::SVM(Problem prob, Kernel k):
            kernel(k),
            problem(prob),
            alpha(prob.size, 0),
            weights(prob.no_dim, 0),
            bias(0)
            {
            }

void SVM::update_weights_if_linear(double a1_new, size_t i1, double a2_new, size_t i2) {
        for (size_t dim = 0; dim < weights.size(); dim ++){
            weights[dim] += problem.labels[i1] * (a1_new - alpha[i1]) * problem.training_input[i1][dim] +
                            problem.labels[i2] * (a2_new - alpha[i2]) * problem.training_input[i2][dim];
        }
    }

void SVM::update_bias(double a1_new, size_t i1, double a2_new, size_t i2){
    double xi1_dot_xi1 = kernel(problem.training_input[i1], problem.training_input[i1]);
    double xi1_dot_xi2 = kernel(problem.training_input[i1], problem.training_input[i2]);
    double xi2_dot_xi2 = kernel(problem.training_input[i2], problem.training_input[i2]);
    double b1 = error[i1] + problem.labels[i1] * (a1_new - alpha[i1]) * xi1_dot_xi1 +
                problem.labels[i2] * (a2_new - alpha[i2]) * xi1_dot_xi2 + bias;
    double b2 = error[i2] + problem.labels[i1] * (a1_new - alpha[i1]) * xi1_dot_xi2 +
                problem.labels[i2] * (a2_new - alpha[i2]) * xi2_dot_xi2 + bias;
    bias = (b1 + b2) / 2;
}

int SVM::take_step(size_t i1, size_t i2){
    double a1 = alpha[i1];
    double a2 = alpha[i2];
    return 1;
    
}

void SVM::SMO() {
    for (size_t idx_alpha = 0; idx_alpha < alpha.size(); idx_alpha++){
    }
}
double SVM::predict(const std::vector<double>& input) {
    double output = -bias;
    for (size_t i = 0; i < input.size(); i++){
        output += weights[i] * input[i];
    }
    return output;
}
int SVM::predict_label(const std::vector<double>& input) {
    double output = -bias;
    for (size_t i = 0; i < input.size(); i++){
        output += weights[i] * input[i];
    }
    return (output > 0 ? 1: -1);
}
