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
        const double tolerance;
        const double epsilon;
        const size_t size;
        const size_t no_dim;
    };
    class SVM{
        public:
            const Problem problem;
            Kernel const* kernel;
            SVM(Problem prob, Kernel const* k);
            void SMO();
        private:
            std::vector<double> alpha;
            std::vector<double> alpha_prime;
            double b_up;
            double b_low;
            size_t i_up;
            size_t i_low;
            std::vector<double> unbiased_error;
            int examineExample(size_t i2);
            int take_step(size_t i1, size_t i2);

        
    };
}
