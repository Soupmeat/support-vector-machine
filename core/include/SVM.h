#pragma once
#include <vector>
#include <functional>

typedef std::function<double(std::vector<double>, std::vector<double>)> Kernel;

double dot(std::vector<double> vector1, std::vector<double> vector2);
inline const Kernel linear = dot;
struct Problem{
    const std::vector<std::vector<double>> training_input;
    const std::vector<int> labels;
    const double C;
    const double tolerance;
    const size_t size;
    const size_t no_dim;
};

class SVM{
public:
    const Problem problem;
    const Kernel kernel;
    double predict(const std::vector<double>& input);
    int predict_label(const std::vector<double>& input);
    void SMO();
    SVM(Problem prob, Kernel k);
private:
    std::vector<double> alpha;
    std::vector<double> weights;
    double bias;
    std::vector<double> error;
    void update_weights_if_linear(double a1_new, size_t i1, double a2_new, size_t i2);
    void update_bias(double a1_new, size_t i1, double a2_new, size_t i2);
    int take_step(size_t i1, size_t i2);
    int examineExample(size_t i2);

};