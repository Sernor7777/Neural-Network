#include "network.hpp"

#include "loss.hpp"
#include "optimizer.hpp"
#include "trainingData.hpp"

#include <iostream>
#include <vector>
#include <algorithm>

Eigen::VectorXd Network::forward(const Eigen::VectorXd& inputs)
{
    Eigen::VectorXd currentInputs = inputs;

    for (Layer& layer : layers)
    {
        currentInputs = layer.forward(currentInputs);
    }

    return currentInputs;
}

Eigen::VectorXd Network::forward(const std::vector<double>& inputs)
{
    Eigen::VectorXd currentInputs = Eigen::Map<const Eigen::VectorXd>(inputs.data(), inputs.size());

    for (Layer& layer : layers)
    {
        currentInputs = layer.forward(currentInputs);
    }

    return currentInputs;
}

void Network::backward(const Eigen::VectorXd& outputGradients)
{
    Eigen::VectorXd gradients = outputGradients;

    for (size_t i = layers.size(); i-- > 0;)
    {
        gradients = layers[i].backward(gradients);
    }
}

void Network::train(const std::vector<TrainingData>& trainingExamples,
                    uint32_t                         batchSize,
                    uint32_t                         epochs,
                    double                           learningRate,
                    OptimizerType                    optimizerType,
                    SchedulerType                    schedulerType,
                    bool                             enableWarmup)
{
    scheduler.setParameters(learningRate, learningRate / 100,
                            epochs * ((trainingExamples.size() + batchSize - 1) / batchSize), schedulerType, enableWarmup);

    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> targets;

    for (const auto& example : trainingExamples)
    {
        Eigen::Map<const Eigen::VectorXd> inputMap(example.inputs.data(), example.inputs.size());

        Eigen::Map<const Eigen::VectorXd> targetMap(example.targets.data(), example.targets.size());

        inputs.emplace_back(inputMap);
        targets.emplace_back(targetMap);
    }

    for (uint32_t epoch = 1; epoch <= epochs; ++epoch)
    {
        double totalLoss = 0;

        for (uint32_t batchStart = 0; batchStart < trainingExamples.size(); batchStart += batchSize)
        {
            uint32_t batchEnd = std::min(batchStart + batchSize, static_cast<uint32_t>(trainingExamples.size()));
            for (uint32_t i = batchStart; i < batchEnd; ++i)
            {
                auto predictions = forward(inputs[i]);

                totalLoss += Loss::forward(predictions, targets[i]);

                Eigen::VectorXd outputGradients = Loss::derivative(predictions, targets[i]);

                backward(outputGradients);
            }
            uint32_t actualBatchSize = batchEnd - batchStart;
            averageGradients(actualBatchSize);
            learningRate = scheduler.getLearningRate(epoch);
            update(learningRate, optimizerType);
        }

        if (epoch % 1000 == 0)
        {
            std::cout << "Epoch: " << epoch << " Average Loss: " << totalLoss / trainingExamples.size()
                      << " Learning Rate: " << learningRate << '\n';
        }
    }
}

void Network::averageGradients(uint32_t batchSize)
{
    for (Layer& layer : layers)
    {
        layer.averageGradients(batchSize);
    }
}

void Network::update(double learningRate, OptimizerType optimizerType)
{
    std::vector<Eigen::MatrixXd*> weights;
    std::vector<Eigen::VectorXd*> biases;
    std::vector<Eigen::MatrixXd*> weightGradients;
    std::vector<Eigen::VectorXd*> biasGradients;

    weights.reserve(layers.size());
    for (Layer& layer : layers)
    {
        weights.push_back(layer.getWeights());
        biases.push_back(layer.getbiases());
        weightGradients.push_back(layer.getWeightGradients());
        biasGradients.push_back(layer.getbiasGradients());
    }

    optimizer.update(weights, weightGradients, biases, biasGradients, learningRate, optimizerType);

    for (Layer& layer : layers)
    {
        layer.zeroGradients();
    }
}

void Network::addLayer(uint32_t inputCount, uint32_t neuronCount, ActivationType activationType)
{
    layers.emplace_back(inputCount, neuronCount, activationType);
    optimizer.addLayer(inputCount, neuronCount);
}

void Network::addLayer(uint32_t neuronCount, ActivationType activationType)
{
    uint32_t inputCount = layers.back().getNeuronCount();
    layers.emplace_back(inputCount, neuronCount, activationType);
    optimizer.addLayer(inputCount, neuronCount);
}