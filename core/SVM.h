#pragma once
#include <vector>
#include <limits>
class SVM {
    private:
        std::vector<std::vector<double>>training_input;
        std::vector<int> labels;
        double C;
        double objective;
        std::vector<double> alpha;
        void step();
    public:
        SVM(std::vector<std::vector<double>> training_inputt, 
            std::vector<int> labelss, double CC);
};
