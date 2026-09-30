#pragma once
#include <vector>
#include <cmath>

enum class KernelType {
    Linear,
    RBF,
    Polynomial
};

double dot(const double* vector1, const double* vector2, size_t size);
struct Kernel{
    const KernelType type;
    virtual double operator()(const double* vector1, const double* vector2, size_t size) = 0;
    Kernel(KernelType t) : type(t) {};
    virtual ~Kernel() = default;
};
struct LinearKernel : public Kernel{
    double operator()(const double* vector1, const double* vector2, size_t size) override;
    LinearKernel(): Kernel(KernelType::Linear) {};
};
struct RBFKernel : public Kernel{
    const double gamma;
    double operator()(const double* vector1, const double* vector2, size_t size) override;
    RBFKernel(double g) : Kernel(KernelType::RBF), gamma(g) {};
};
struct PolynomialKernel : public Kernel{
    const double degree;
    const double bias;
    PolynomialKernel(double d, double b) : Kernel(KernelType::Polynomial), degree(d), bias(b) {};
    double operator()(const double* vector1, const double* vector2, size_t size) override;
};