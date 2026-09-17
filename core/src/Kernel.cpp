#include <Kernel.h>

double dot(std::vector<double> vector1, std::vector<double> vector2)
{
    size_t size = vector1.size();
    double answer = 0;
    for (size_t idx = 0; idx < size; idx++)
    {
        answer += vector1[idx] * vector2[idx];
    }
    return answer;
}

double LinearKernel::operator()(const std::vector<double>& vector1, const std::vector<double>& vector2)
    {
        return dot(vector1, vector2);
    }
double RBFKernel::operator()(const std::vector<double>& vector1, const std::vector<double>& vector2)
    {
        double sum = 0;
        for (size_t i = 0; i < vector1.size(); i++)
        {
            double diff = vector1[i] - vector2[i];
            sum += diff * diff;
        }
        return exp(-gamma * sum);
    }
double PolynomialKernel::operator()(const std::vector<double>& vector1, const std::vector<double>& vector2)
    {
        return pow(dot(vector1, vector2) + bias, degree);
    }