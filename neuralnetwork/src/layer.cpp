#include "layer.hpp"

#include <random>

Layer::Layer(uint32_t inputCount, uint32_t neuronCount, ActivationType activationType) : activationType(activationType)
{
    double limit = 0;
    switch (activationType)
    {
    case ActivationType::ReLU :
        limit = std::sqrt(6.0 / inputCount);  // He Uniform
        break;
    default :
        limit = std::sqrt(6.0 / (inputCount + neuronCount));  // Xavier Uniform
    }

    std::random_device randomDevice;
    std::mt19937       generator(randomDevice());
    // std::mt19937                           generator(234);
    std::uniform_real_distribution<double> distribution(-limit, limit);

    weights.resize(neuronCount, inputCount);
    for (Eigen::Index row = 0; row < weights.rows(); ++row)
    {
        for (Eigen::Index col = 0; col < weights.cols(); ++col)
        {
            weights(row, col) = distribution(generator);
        }
    }

    weightGradients.resize(neuronCount, inputCount);
    weightGradients.setZero();
    biases.resize(neuronCount);
    biases.setZero();
    biasGradients.resize(neuronCount);
    biasGradients.setZero();
}

Eigen::VectorXd Layer::forward(const Eigen::VectorXd& inputs)
{
    lastInputs = inputs;
    lastZs     = weights * inputs + biases;
    return activation(lastZs);
}

Eigen::VectorXd Layer::backward(const Eigen::VectorXd& outputGradients)
{
    dA_dZ = activationDerivative(lastZs);

    dL_dZ = outputGradients.array() * dA_dZ.array();

    weightGradients += dL_dZ * lastInputs.transpose();
    biasGradients += dL_dZ;

    return weights.transpose() * dL_dZ;
}

void Layer::averageGradients(size_t batchSize)
{
    weightGradients /= batchSize;
    biasGradients /= batchSize;
}
void Layer::zeroGradients()
{
    weightGradients.setZero();
    biasGradients.setZero();
}

Eigen::VectorXd Layer::activation(const Eigen::VectorXd& x)
{
    Eigen::VectorXd output(x.size());

    for (Eigen::Index i = 0; i < x.size(); ++i)
    {
        switch (activationType)
        {
        case ActivationType::ReLU :
            output[i] = x[i] < 0 ? 0 : x[i];
            break;
        case ActivationType::Sigmoid :
            output[i] = 1 / (1 + std::exp(-x[i]));
            break;
        case ActivationType::Linear :
            output[i] = x[i];
            break;
        case ActivationType::Tanh :
            output[i] = std::tanh(x[i]);
            break;
        }
    }

    return output;
}

Eigen::VectorXd Layer::activationDerivative(const Eigen::VectorXd& x)
{
    Eigen::VectorXd output(x.size());

    for (Eigen::Index i = 0; i < x.size(); ++i)
    {
        switch (activationType)
        {
        case ActivationType::ReLU :
            output[i] = x[i] < 0 ? 0 : 1;
            break;
        case ActivationType::Sigmoid : {
            double sigmoid = 1 / (1 + std::exp(-x[i]));
            output[i]      = sigmoid * (1 - sigmoid);
            break;
        }
        case ActivationType::Linear :
            output[i] = 1;
            break;
        case ActivationType::Tanh : {
            double tanh = std::tanh(x[i]);
            output[i]   = 1 - (tanh * tanh);
            break;
        }
        }
    }

    return output;
}