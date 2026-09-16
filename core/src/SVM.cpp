#include <SVM.h>
#include <vector>
#include <random>
#define abs(x) (((x) < 0) ? -(x) : (x))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))
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
    double eps = problem.eps;
    if (a1_new > eps && a1_new < problem.C - eps)
        bias = b1;
    else if (a2_new > eps && a2_new < problem.C - eps)
        bias = b2;
    else
        bias = (b1 + b2) / 2;
}

int SVM::take_step(size_t i1, size_t i2)
{
    if (i1 == i2)
        return 0;
    double alph1 = alpha[i1];
    double alph2 = alpha[i2];
    double y1 = problem.labels[i1];
    double y2 = problem.labels[i2];
    double E1 = error[i1];
    double E2 = error[i2];
    double s = y1 * y2;
    double eps = problem.eps;
    std::pair<double, double> L_H = compute_L_H(alph1, alpha[i2], y1, problem.labels[i2]);
    double L = L_H.first;
    double H = L_H.second;
    if (L == H)
        return 0;
    double k11 = kernel(problem.training_input[i1], problem.training_input[i1]);
    double k12 = kernel(problem.training_input[i1], problem.training_input[i2]);
    double k22 = kernel(problem.training_input[i2], problem.training_input[i2]);
    double eta = k11 + k22 - 2 * k12;
    double a2;
    double old_bias = bias;
    if (eta > 0)
    {
        a2 = alph2 + y2 * (E1 - E2) / eta;
        if (a2 < L)
            a2 = L;
        else if (a2 > H)
            a2 = H;
    }
    else
    {
        double f_1 = y1 * (E1 + bias) - alph1 * k11 - s * alph2 * k12;
        double f_2 = y2 * (E2 + bias) - s * alph1 * k12 - alph2 * k22;
        double L1 = alph1 + s * (alph2 - L);
        double H1 = alph1 + s * (alph2 - H);
        double Lobj = L1 * f_1 + L * f_2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + s * L * L1 * k12;
        double Hobj = H1 * f_1 + H * f_2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + s * H * H1 * k12;

        if (Lobj < Hobj - eps)
            a2 = L;
        else if (Lobj > Hobj + eps)
            a2 = H;
        else
            a2 = alph2;
    }
    if (abs(a2 - alph2) < eps * (a2 + alph2 + eps))
        return 0;
    double a1 = max(0, alph1 + s * (alph2 - a2));
    if ((alph1 > eps && alph1 < problem.C - eps) && !(a1 > eps && a1 < problem.C - eps))
    {
        nonbound--;
        isnonbound[i1] = false;
    }
    else if (!(alph1 > eps && alph1 < problem.C - eps) && (a1 > eps && a1 < problem.C - eps))
    {
        nonbound++;
        isnonbound[i1] = true;
    }
    if ((alph2 > eps && alph2 < problem.C - eps) && !(a2 > eps && a2 < problem.C - eps))
    {
        nonbound--;
        isnonbound[i2] = false;
    }
    else if (!(alph2 > eps && alph2 < problem.C - eps) && (a2 > eps && a2 < problem.C - eps))
    {
        nonbound++;
        isnonbound[i2] = true;
    }
    update_bias(a1, i1, a2, i2);
    if (kernel.type == KernelType::Linear)
        update_weights_if_linear(a1, i1, a2, i2);
    update_errors(a1 - alph1, a2 - alph2, bias - old_bias, i1, i2);
    alpha[i1] = a1;
    alpha[i2] = a2;
    return 1;
}

void SVM::SMO()
{
    int num_changed = 0;
    int examine_all = 1;
    while (num_changed > 0 || examine_all)
    {
        num_changed = 0;
        if (examine_all)
        {
            for (size_t i = 0; i < alpha.size(); i++)
            {
                num_changed += examineExample(i);
            }
        }
        else
        {
            for (size_t i = 0; i < alpha.size(); i++)
            {
                if (isnonbound[i])
                {
                    num_changed += examineExample(i);
                }
            }
        }
        if (examine_all == 1)
            examine_all = 0;
        else if (num_changed == 0)
            examine_all = 1;
    }
}

int SVM::examineExample(size_t i2)
{
    int y2 = problem.labels[i2];
    double alph2 = alpha[i2];
    double E2 = error[i2];
    double r2 = E2 * y2;
    double tol = problem.tolerance;
    double C = problem.C;
    double eps = problem.eps;
    if ((r2 < -tol && alph2 < C - eps) || (r2 > tol && alph2 > eps))
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
        if (nonbound > 1)
        {
            for (size_t k = 0; k < alpha.size(); k++)
            {
                size_t i1 = (random_idx + k * stride) % alpha.size();
                if ((not isnonbound[i1]) || (i1 == i2))
                    continue;
                if (take_step(i1, i2))
                    return 1;
            }
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
    return (output > problem.eps ? 1 : -1);
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

std::pair<double, double> SVM::compute_L_H(double a1, double a2, int y1, int y2)
{
    double L = 0;
    double H = 0;
    if (y1 != y2)
    {
        L = max(0.0, a2 - a1);
        H = min(problem.C, problem.C + a2 - a1);
    }
    else
    {
        L = max(0.0, a2 + a1 - problem.C);
        H = min(problem.C, a2 + a1);
    }
    return {L, H};
}