#pragma once

#include <Eigen/Dense>

#include <cstdint>

enum class ActivationType
{
    ReLU,
    Sigmoid,
    Linear,
    Tanh
};

class Layer
{
public:
    Layer(uint32_t inputCount, uint32_t neuronCount, ActivationType activationType);

    Eigen::VectorXd forward(const Eigen::VectorXd& inputs);
    Eigen::VectorXd backward(const Eigen::VectorXd& outputGradients);

    void averageGradients(size_t batchSize);
    void zeroGradients();

    [[nodiscard]] Eigen::MatrixXd* getWeights() { return &weights; }
    [[nodiscard]] Eigen::VectorXd* getbiases() { return &biases; }
    [[nodiscard]] Eigen::MatrixXd* getWeightGradients() { return &weightGradients; }
    [[nodiscard]] Eigen::VectorXd* getbiasGradients() { return &biasGradients; }

    [[nodiscard]] uint32_t getNeuronCount() const { return biases.size(); }

private:
    ActivationType activationType;

    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;

    Eigen::MatrixXd weightGradients;
    Eigen::VectorXd biasGradients;

    Eigen::VectorXd lastInputs;
    Eigen::VectorXd lastZs;

    Eigen::VectorXd dA_dZ;
    Eigen::VectorXd dL_dZ;

    Eigen::VectorXd activation(const Eigen::VectorXd& x);
    Eigen::VectorXd activationDerivative(const Eigen::VectorXd& x);
};