#include <SVR.h>
using namespace SVR;
#define abs(x) (((x) < 0) ? -(x) : (x))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))
SVM::SVM(Problem prob, Kernel const* k) : kernel(k),
                                                       problem(prob),
                                                       alpha(prob.size, 0),
                                                       alpha_prime(prob.size, 0),
                                                       unbiased_error(prob.size)
{
    double max = double(-INFINITY); double min = double(INFINITY);
    size_t max_idx = 0; size_t min_idx = 0;
    for (size_t idx = 0; idx < unbiased_error.size(); idx++)
    {
        unbiased_error[idx] = problem.targets[idx];
        if (max < unbiased_error[idx])
        {
            max = unbiased_error[idx];
            max_idx = idx;
        }
        if (min > unbiased_error[idx])
        {
            min = unbiased_error[idx];
            min_idx = idx;
        }
    }
    b_up = min + problem.epsilon;
    b_low = max - problem.epsilon;
    i_up = min_idx;
    i_low = max_idx;
}

void SVM::SMO()
{
    int examine_all = 1;
    int num_changed = 0;
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
                //implement check for some criteria
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