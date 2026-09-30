#include <Kernel.h>
#include <omp.h>
double dot(const double* vector1, const double* vector2, size_t size)
{
    double answer = 0;
    #pragma omp simd reduction(+:answer)
    for (size_t idx = 0; idx < size; idx++)
    {
        answer += vector1[idx] * vector2[idx];
    }
    return answer;
}

double LinearKernel::operator()(const double* vector1, const double* vector2, size_t size)
    {
        return dot(vector1, vector2, size);
    }
double RBFKernel::operator()(const double* vector1, const double* vector2, size_t size)
    {
        double sum = 0;
        #pragma omp simd reduction(+:sum)
        for (size_t i = 0; i < size; i++)
        {
            double diff = vector1[i] - vector2[i];
            sum += diff * diff;
        }
        return exp(-gamma * sum);
    }

double PolynomialKernel::operator()(const double* vector1, const double* vector2, size_t size)
    {
        return pow(dot(vector1, vector2, size) + bias, degree);
    }
