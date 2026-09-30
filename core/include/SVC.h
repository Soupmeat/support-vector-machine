#pragma once
#include <vector>
#include <random>
#include <Kernel.h>
#include <span>
namespace SVC
{
    struct Problem
    {
        const double* training_input;
        const int* labels;
        const double C;
        const double tolerance;
        const size_t size;
        const size_t no_dim;
        const double eps;
        const int max_iter = 1000;

        double get(size_t row, size_t index) const
        {
            return training_input[row * no_dim + index];
        }
        double get_label(size_t row) const
        {
            return labels[row];
        }
        std::span<const double> get_row(size_t row) const
        {
            return std::span<const double>(training_input + (row * no_dim), no_dim);
        }
    };

    class SVM
    {
    public:
        const Problem problem;
        Kernel *const kernel;
        double predict(const double* input);
        int predict_label(const double* input);
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