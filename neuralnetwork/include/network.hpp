#pragma once

#include "layer.hpp"
#include "optimizer.hpp"
#include "scheduler.hpp"
#include "trainingData.hpp"

#include <vector>
#include <cstdint>

class Network
{
public:
    Eigen::VectorXd forward(const Eigen::VectorXd& inputs);
    Eigen::VectorXd forward(const std::vector<double>& inputs);
    void            backward(const Eigen::VectorXd& outputGradients);

    void train(const std::vector<TrainingData>& trainingExamples,
               uint32_t                         batchSize,
               uint32_t                         epochs,
               double                           learningRate,
               OptimizerType                    optimizerType,
               SchedulerType                    schedulerType,
               bool                             enableWarmup = false);

    void averageGradients(uint32_t batchSize);
    void update(double learningRate, OptimizerType optimizerType);

    void addLayer(uint32_t inputCount, uint32_t neuronCount, ActivationType activationType);
    void addLayer(uint32_t neuronCount, ActivationType activationType);

private:
    std::vector<Layer> layers;

    Optimizer optimizer;
    Scheduler scheduler;
};