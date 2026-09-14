#pragma once

#include <Eigen/Dense>

class Loss  // Probably change to a namespace later.
{
public:
    static double          forward(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets);
    static Eigen::VectorXd derivative(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets);

private:
};