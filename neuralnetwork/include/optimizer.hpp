#pragma once

#include <Eigen/Dense>

#include <vector>

enum class OptimizerType
{
    SGD,
    Adam
};

class Optimizer
{
public:
    void update(std::vector<Eigen::MatrixXd*>& weights,
                std::vector<Eigen::MatrixXd*>& weightGradients,
                std::vector<Eigen::VectorXd*>& biases,
                std::vector<Eigen::VectorXd*>& biasGradients,
                double                         learningRate,
                OptimizerType                  optimizerType);

    void addLayer(uint32_t inputCount, uint32_t neuronCount);

private:
    std::vector<Eigen::MatrixXd> weightFirstMoments;
    std::vector<Eigen::MatrixXd> weightSecondMoments;
    std::vector<Eigen::VectorXd> biasFirstMoments;
    std::vector<Eigen::VectorXd> biasSecondMoments;

    double beta1Power = 1;
    double beta2Power = 1;

    std::vector<Eigen::MatrixXd> biasCorrectedWeightFirst;
    std::vector<Eigen::MatrixXd> biasCorrectedWeightSecond;
    std::vector<Eigen::VectorXd> biasCorrectedBiasFirst;
    std::vector<Eigen::VectorXd> biasCorrectedBiasSecond;
};