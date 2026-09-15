#include <SVM.h>
#include <vector>
#include <random>
#define abs(x) (((x) < 0) ? -(x) : (x))
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
SVM::SVM(Problem prob, Kernel k, unsigned int seed) : kernel(k),
                                                      problem(prob),
                                                      alpha(prob.size, 0),
                                                      weights(prob.no_dim, 0),
                                                      bias(0),
                                                      isnonbound(prob.size, false),
                                                      error(prob.size),
                                                      nonbound(0),
                                                      rng(seed)
{
    for (size_t idx = 0; idx < error.size(); idx++)
    {
        error[idx] = -problem.labels[idx];
    }
}

void SVM::update_weights_if_linear(double a1_new, size_t i1, double a2_new, size_t i2)
{
    for (size_t dim = 0; dim < weights.size(); dim++)
    {
        weights[dim] += problem.labels[i1] * (a1_new - alpha[i1]) * problem.training_input[i1][dim] +
                        problem.labels[i2] * (a2_new - alpha[i2]) * problem.training_input[i2][dim];
    }
}

void SVM::update_bias(double a1_new, size_t i1, double a2_new, size_t i2)
{
    double xi1_dot_xi1 = kernel(problem.training_input[i1], problem.training_input[i1]);
    double xi1_dot_xi2 = kernel(problem.training_input[i1], problem.training_input[i2]);
    double xi2_dot_xi2 = kernel(problem.training_input[i2], problem.training_input[i2]);
    double b1 = error[i1] + problem.labels[i1] * (a1_new - alpha[i1]) * xi1_dot_xi1 +
                problem.labels[i2] * (a2_new - alpha[i2]) * xi1_dot_xi2 + bias;
    double b2 = error[i2] + problem.labels[i1] * (a1_new - alpha[i1]) * xi1_dot_xi2 +
                problem.labels[i2] * (a2_new - alpha[i2]) * xi2_dot_xi2 + bias;
    bias = (b1 + b2) / 2;
}

int SVM::take_step(size_t i1, size_t i2)
{
    double a1 = alpha[i1];
    double a2 = alpha[i2];
    return 1;
}

void SVM::SMO()
{
}

int SVM::examineExample(size_t i2)
{
    int y2 = problem.labels[i2];
    double alph2 = alpha[i2];
    double E2 = error[i2];
    double r2 = E2 * y2;
    double tol = problem.tolerance;
    double C = problem.C;
    if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))
    {
        if (nonbound > 1)
        {
            size_t i1 = select_optimal_i1(i2);
            if (take_step(i1, i2))
                return 1;
        }
        std::uniform_int_distribution<size_t> dist(0, alpha.size() - 1);

        size_t random_idx = dist(rng);
        constexpr size_t stride = 100000007; // assume that the number of samples is less than 100 million
        for (size_t k = 0; k < alpha.size(); k++)
        {
            size_t i1 = (random_idx + k * stride) % alpha.size();
            if ((not isnonbound[i1]) || (i1 == i2))
                continue;
            if (take_step(i1, i2))
                return 1;
        }
        for (size_t k = 0; k < alpha.size(); k++)
        {
            size_t i1 = (random_idx + k * stride) % alpha.size();
            if (i1 == i2)
                continue;
            if (take_step(i1, i2))
                return 1;
        }
    }
    return 0;
}

size_t SVM::select_optimal_i1(size_t i2)
{
    double max_error = 0;
    size_t optimal_i1 = i2;
    for (size_t i1 = 0; i1 < alpha.size(); i1++)
    {
        if ((i1 == i2) || (not isnonbound[i1]))
        {
            continue;
        }
        double Error = abs(error[i1] - error[i2]);
        if (Error > max_error)
        {
            max_error = Error;
            optimal_i1 = i1;
        }
    }
    return optimal_i1;
}

double SVM::predict(const std::vector<double> &input)
{
    double output = -bias;
    for (size_t i = 0; i < input.size(); i++)
    {
        output += weights[i] * input[i];
    }
    return output;
}
int SVM::predict_label(const std::vector<double> &input)
{
    double output = -bias;
    for (size_t i = 0; i < input.size(); i++)
    {
        output += weights[i] * input[i];
    }
    return (output > 0 ? 1 : -1);
}

void SVM::update_errors(double diff_a1, double diff_a2, double diff_b, size_t i1, size_t i2)
{
    for (size_t idx = 0; idx < error.size(); idx++)
    {
        error[idx] += problem.labels[i1] *
                          diff_a1 *
                          kernel(problem.training_input[idx], problem.training_input[i1]) +
                      problem.labels[i2] *
                          diff_a2 *
                          kernel(problem.training_input[idx], problem.training_input[i2]) +
                      diff_b;
    }
}
