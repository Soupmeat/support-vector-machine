#include <SVR.h>
using namespace SVR;
#define abs(x) (((x) < 0) ? -(x) : (x))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))
SVM::SVM(Problem prob, Kernel * k) : kernel(k),
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

void SVM::update_error(size_t i1, size_t i2, double delta_i1)
{
    for (size_t idx = 0; idx < beta.size(); idx++)
    {
        unbiased_error[idx] -= delta_i1 * ((*kernel)(problem.training_input[i1], problem.training_input[idx]) - (*kernel)(problem.training_input[i2], problem.training_input[idx]));
    }
}

int SVM::take_step(size_t i1, size_t i2, double beta_i1, double beta_i2) {
    if (i1 == i2) {
        return 0;
    }

    double K_11 = (*kernel)(problem.training_input[i1], problem.training_input[i1]);
    double K_22 = (*kernel)(problem.training_input[i2], problem.training_input[i2]);
    double K_12 = (*kernel)(problem.training_input[i1], problem.training_input[i2]);

    double eta = K_11 + K_22 - 2.0 * K_12;

    double gamma = beta_i1 + beta_i2;
    
    double L = max(-problem.C, gamma - problem.C);
    double H = min(problem.C, gamma + problem.C);

    if (abs(L - H) <= problem.mu * (L + H + problem.mu)) {
        return 0;
    }

    double gap = error_low[i1] - error_up[i2];

    double beta_i1_new = 0.0;

    if (eta > problem.mu) {
        double delta = gap / eta;
        double beta_i_unc = beta_i1 + delta;

        beta_i1_new = max(L, min(H, beta_i_unc));
    } else {
        double obj_L = L * gap - 0.5 * eta * (L * L);
        double obj_H = H * gap - 0.5 * eta * (H * H);

        if (obj_L > obj_H + problem.mu) {
            beta_i1_new = L;
        } else if (obj_H > obj_L + problem.mu) {
            beta_i1_new = H;
        } else {
            return 0;
        }
    }

    double d_i = beta_i1_new - beta_i1;

    if (abs(d_i) < problem.mu * (abs(beta_i1_new) + abs(beta_i1) + problem.mu)) {
        return 0;
    }

    double beta_i2_new = beta_i2 - d_i;

    beta[i1] = beta_i1_new;
    beta[i2] = beta_i2_new;

    return 1;
}

double SVM::predict(const std::vector<double>& input) const {
    double sum = 0.0;

    for (size_t i = 0; i < problem.training_input.size(); ++i) {
        sum += beta[i] * (*kernel)(problem.training_input[i], input);
    }

    double b = (b_low + b_up) / 2.0;

    return sum + b;
}
int SVM::examine_example(size_t i2) {
    return 0;
}

void SVM::SMO() {
    int examineAll = 0;
    int numChanged = 0;

    do {
        numChanged = 0;

        if (examineAll) {
            for (size_t i = 0; i < problem.training_input.size(); ++i) {
                numChanged += examine_example(i);
            }
        } else {
            while (b_low > b_up + 2.0 * problem.tolerance) {
                size_t i1 = i_low;
                size_t i2 = i_up;
                double beta_i1_old = beta[i1];

                if (take_step(i1, i2, beta[i1], beta[i2])) {
                    double d_i = beta[i1] - beta_i1_old;
                    update_error(i1, i2, beta[i1] - beta_i1_old);
                    update_error_up_low();
                    update_bounds();
                    numChanged++;
                } else {
                    break;
                }
            }
        }

        if (examineAll == 1) {
            examineAll = 0;
        } else if (numChanged == 0) {
            examineAll = 1;
        }
    } while (numChanged > 0 || examineAll);
}