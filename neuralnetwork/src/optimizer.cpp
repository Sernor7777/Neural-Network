#include "optimizer.hpp"

void Optimizer::update(std::vector<Eigen::MatrixXd*>& weights,
                       std::vector<Eigen::MatrixXd*>& weightGradients,
                       std::vector<Eigen::VectorXd*>& biases,
                       std::vector<Eigen::VectorXd*>& biasGradients,
                       double                         learningRate,
                       OptimizerType                  optimizerType)
{
    switch (optimizerType)
    {
    case OptimizerType::Adam :
        beta1Power *= 0.9;
        beta2Power *= 0.999;

        for (size_t i = 0; i < weights.size(); ++i)
        {
            weightFirstMoments[i]  = 0.9 * weightFirstMoments[i] + 0.1 * *weightGradients[i];
            weightSecondMoments[i] = 0.999 * weightSecondMoments[i] + 0.001 * weightGradients[i]->array().square().matrix();

            biasFirstMoments[i]  = 0.9 * biasFirstMoments[i] + 0.1 * *biasGradients[i];
            biasSecondMoments[i] = 0.999 * biasSecondMoments[i] + 0.001 * biasGradients[i]->array().square().matrix();

            biasCorrectedWeightFirst[i]  = weightFirstMoments[i] / (1 - beta1Power);
            biasCorrectedWeightSecond[i] = weightSecondMoments[i] / (1 - beta2Power);
            biasCorrectedBiasFirst[i]    = biasFirstMoments[i] / (1 - beta1Power);
            biasCorrectedBiasSecond[i]   = biasSecondMoments[i] / (1 - beta2Power);

            weights[i]->array() -=
                learningRate * biasCorrectedWeightFirst[i].array() / (biasCorrectedWeightSecond[i].array().sqrt() + 1e-08);
            biases[i]->array() -=
                learningRate * biasCorrectedBiasFirst[i].array() / (biasCorrectedBiasSecond[i].array().sqrt() + 1e-08);
        }
        break;
    default : {
        for (size_t i = 0; i < weights.size(); ++i)
        {
            *weights[i] -= learningRate * *weightGradients[i];
            *biases[i] -= learningRate * *biasGradients[i];
        }
        break;
    }
    }
}

void Optimizer::setParameters(std::vector<uint32_t> inputCount,
                              std::vector<uint32_t> neuronCount,
                              OptimizerType         optimizerType)
{
    size_t layerCount = inputCount.size();

    weightFirstMoments.resize(layerCount);
    weightSecondMoments.resize(layerCount);
    biasFirstMoments.resize(layerCount);
    biasSecondMoments.resize(layerCount);

    for (size_t i = 0; i < layerCount; ++i)
    {
        weightFirstMoments[i].setZero(neuronCount[i], inputCount[i]);
        weightSecondMoments[i].setZero(neuronCount[i], inputCount[i]);

        biasFirstMoments[i].setZero(neuronCount[i]);
        biasSecondMoments[i].setZero(neuronCount[i]);
    }
}

void Optimizer::addLayer(uint32_t inputCount, uint32_t neuronCount)
{
    weightFirstMoments.emplace_back(Eigen::MatrixXd::Zero(neuronCount, inputCount));
    weightSecondMoments.emplace_back(Eigen::MatrixXd::Zero(neuronCount, inputCount));
    biasFirstMoments.emplace_back(Eigen::VectorXd::Zero(neuronCount));
    biasSecondMoments.emplace_back(Eigen::VectorXd::Zero(neuronCount));

    biasCorrectedWeightFirst.emplace_back(Eigen::MatrixXd::Zero(neuronCount, inputCount));
    biasCorrectedWeightSecond.emplace_back(Eigen::MatrixXd::Zero(neuronCount, inputCount));
    biasCorrectedBiasFirst.emplace_back(Eigen::VectorXd::Zero(neuronCount));
    biasCorrectedBiasSecond.emplace_back(Eigen::VectorXd::Zero(neuronCount));
}