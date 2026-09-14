#include <network.hpp>

#include <iostream>
#include <numbers>
#include <cmath>
#include <chrono>
#include <random>

int main()
{
    std::vector<TrainingData> trainingExamples;

    const int sampleCount = 100;

    for (int i = 0; i < sampleCount; ++i)
    {
        double x = -std::numbers::pi + ((2.0 * std::numbers::pi * i) / (sampleCount - 1));
        double y = std::sin(x);

        trainingExamples.emplace_back(TrainingData{.inputs = {x}, .targets = {y}});
    }

    Network network;
    network.addLayer(1, 10, ActivationType::Tanh);
    network.addLayer(10, ActivationType::Tanh);
    network.addLayer(1, ActivationType::Linear);

    uint32_t epochs = 100000;

    auto start = std::chrono::steady_clock::now();

    network.train(trainingExamples, 100, epochs, 0.0003, OptimizerType::Adam, SchedulerType::Cosine_Decay, true);

    auto   end     = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Training took " << seconds << " seconds\n";
    std::cout << "Time per epoch: " << seconds / epochs * 1000.0 << " ms\n";

    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_real_distribution<double> dist(-std::numbers::pi, std::numbers::pi);
    double                                 totalError = 0;
    for (int i = 0; i < 10; ++i)
    {
        double          x          = dist(gen);
        double          actual     = std::sin(x);
        Eigen::VectorXd output     = network.forward(std::vector<double>{x});
        double          prediction = output[0];
        totalError += std::abs(prediction - actual);
        std::cout << "x: " << x << " | Prediction: " << prediction << " | Actual: " << actual
                  << " | Error: " << prediction - actual << '\n';
    }
    std::cout << "Average Error: " << totalError / 10 << '\n';
}