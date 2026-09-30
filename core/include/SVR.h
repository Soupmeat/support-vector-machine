#pragma once
#include <vector>
#include <random>
#include <Kernel.h>

namespace SVR{
    struct Problem
    {
        const double* training_input;
        const double* targets;
        const double C;
        const double tolerance; // for stopping criteria
        const double epsilon;
        const size_t size;
        const size_t no_dim;
        const double mu; // threshold for floating point comparison
        const int max_iter = 1000;
        
        double get(size_t row, size_t index) const
        {
            return training_input[row * no_dim + index];
        }
        double get_target(size_t row) const
        {
            return targets[row];
        }
        std::span<const double> get_row(size_t row) const
        {
            return std::span<const double>(training_input + (row * no_dim), no_dim);
        }
    };
    class SVM{
        public:
            const Problem problem;
            Kernel *const kernel;
            SVM(Problem prob, Kernel * k);
            void SMO();
            double predict(const double* input) const;
        private:
            std::vector<double> beta;
            std::vector<double> unbiased_error;
            std::vector<double> error_up;
            std::vector<double> error_low;
            size_t i_up;
            size_t i_low;
            double b_up;
            double b_low;
            int take_step(size_t i1, size_t i2, double beta_i1, double beta_i2);
            int examine_example(size_t i2);
            void update_error(size_t i1, size_t i2, double delta_i1);
            void update_error_up_low();
            void update_bounds();
            bool is_non_bound(size_t i) const;
        
    };
}
