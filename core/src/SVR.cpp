#include <SVR.h>
using namespace SVR;
#define abs(x) (((x) < 0) ? -(x) : (x))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))
SVM::SVM(Problem prob, Kernel const *k) : kernel(k),
                                          problem(prob),
                                          beta(prob.size, 0.0),
                                          unbiased_error(prob.size),
                                          error_up(prob.size),
                                          error_low(prob.size)
{
    for (size_t i = 0; i < problem.size; i++)
    {
        unbiased_error[i] = problem.targets[i];
    }
    update_error_up_low();
    update_bounds();
}

void SVM::SVM::update_error_up_low()
{
    for (size_t idx = 0; idx < beta.size(); idx++)
    {
        if (beta[idx] > problem.mu)
            error_low[idx] = unbiased_error[idx] - problem.epsilon;
        else if (beta[idx] < -problem.mu && beta[idx] > -problem.C)
            error_low[idx] = unbiased_error[idx] + problem.epsilon;
        else if (beta[idx] <= problem.mu && beta[idx] >= -problem.mu)
        {
            error_low[idx] = unbiased_error[idx] - problem.epsilon;
        }
        else
        {
            error_low[idx] = double(-INFINITY);
        }
    }
    for (size_t idx = 0; idx < beta.size(); idx++)
    {
        if (beta[idx] > problem.mu && beta[idx] < problem.C)
            error_up[idx] = unbiased_error[idx] - problem.epsilon;
        else if (beta[idx] <= problem.mu)
            error_up[idx] = unbiased_error[idx] + problem.epsilon;
        else
        {
            error_up[idx] = double(INFINITY);
        }
    }
}

void SVM::update_bounds()
{
    b_up = double(INFINITY);
    b_low = double(-INFINITY);
    for (size_t idx = 0; idx < beta.size(); idx++)
    {
        if (error_up[idx] < b_up)
        {
            b_up = error_up[idx];
            i_up = idx;
        }
        if (error_low[idx] > b_low)
        {
            b_low = error_low[idx];
            i_low = idx;
        }
    }
}