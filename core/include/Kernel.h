#pragma once
#include <vector>
#include <cmath>
#include <span>


inline std::span<const double> ROW_SPAN(const std::vector<double>& vec, size_t i, size_t num_dim) {
    return std::span<const double>(vec.data() + (i * num_dim), num_dim);
}

enum class KernelType {
    Linear,
    RBF,
    Polynomial
};

double dot(std::vector<double> vector1, std::vector<double> vector2);
double dot(std::span<const double> vector1, std::span<const double> vector2);

struct Kernel{
    const KernelType type;
    virtual double operator()(std::span<const double> vector1, std::span<const double> vector2) = 0;
    virtual double operator()(const std::vector<double>& vector1, const std::vector<double>& vector2) = 0;
    Kernel(KernelType t) : type(t) {};
    virtual ~Kernel() = default;
};
struct LinearKernel : public Kernel{
    double operator()(const std::vector<double>& vector1, const std::vector<double>& vector2) override;
    double operator()(std::span<const double> vector1, std::span<const double> vector2) override;
    LinearKernel(): Kernel(KernelType::Linear) {};
};
struct RBFKernel : public Kernel{
    const double gamma;
    double operator()(std::span<const double> vector1, std::span<const double> vector2) override;
    double operator()(const std::vector<double>& vector1, const std::vector<double>& vector2) override;
    RBFKernel(double g) : Kernel(KernelType::RBF), gamma(g) {};
};
struct PolynomialKernel : public Kernel{
    const double degree;
    const double bias;
    PolynomialKernel(double d, double b) : Kernel(KernelType::Polynomial), degree(d), bias(b) {};
    double operator()(std::span<const double> vector1, std::span<const double> vector2) override;
    double operator()(const std::vector<double>& vector1, const std::vector<double>& vector2) override;
};