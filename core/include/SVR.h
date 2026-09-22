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
        const double eps;
        const size_t size;
        const size_t no_dim;
    };
    class SVM{
        public:
            Problem problem;
            Kernel const* kernel;
        
    };
}
