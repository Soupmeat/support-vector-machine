#pragma once
#include <vector>
#include <random>
#include <Kernel.h>
namespace SVC
{
    struct Problem
    {
        const std::vector<std::vector<double>> training_input;
        const std::vector<int> labels;
        const double C;
        const double tolerance;
        const size_t size;
        const size_t no_dim;
        const double eps;
    };

    class SVM
    {
    public:
        const Problem problem;
        Kernel *const kernel;
        double predict(const std::vector<double> &input);
        int predict_label(const std::vector<double> &input);
        void SMO();
        SVM(Problem prob, Kernel *k, unsigned int seed = 42);

    private:
        std::vector<double> alpha;
        std::vector<double> weights; // only used for linear kernel
        double bias;
        std::vector<double> error;
        std::vector<bool> isnonbound;
        size_t nonbound;
        std::mt19937 rng;
        void update_weights_if_linear(double a1_new, size_t i1, double a2_new, size_t i2);
        void update_bias(double a1_new, size_t i1, double a2_new, size_t i2);
        int take_step(size_t i1, size_t i2);
        int examineExample(size_t i2);
        void update_errors(double diff_a1, double diff_a2, double diff_b, size_t i1, size_t i2);
        size_t select_optimal_i1(size_t i2);
        std::pair<double, double> compute_L_H(double a1, double a2, int y1, int y2);
    };
}