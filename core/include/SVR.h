#pragma once
#include <vector>
#include <random>
#include <Kernel.h>

namespace SVR{
    struct Problem
    {
        const std::vector<std::vector<double>> training_input;
        const std::vector<double> targets;
        const double C;
        const double tolerance; // for stopping criteria
        const double epsilon;
        const size_t size;
        const size_t no_dim;
        const double mu; // threshold for floating point comparison
    };
    class SVM{
        public:
            const Problem problem;
            Kernel const* kernel;
            SVM(Problem prob, Kernel const* k);
            void SMO();
        private:
            std::vector<double> beta;
            std::vector<double> unbiased_error;
            std::vector<double> error_up;
            std::vector<double> error_low;
            size_t i_up;
            size_t i_low;
            double b_up;
            double b_low;
            int take_step(size_t i1, size_t i2);
            void update_error_up_low();
            void update_bounds();
        
    };
}
