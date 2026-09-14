#include "loss.hpp"

double Loss::forward(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets)
{
    return (predictions - targets).squaredNorm() / predictions.size();
}

Eigen::VectorXd Loss::derivative(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets)
{
    return 2 * (predictions - targets);
}