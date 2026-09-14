#include "scheduler.hpp"

#include <cmath>
#include <numbers>

[[nodiscard]] double Scheduler::getLearningRate(uint32_t step) const
{
    if (warmupEnabled && step < warmupSteps) { return initialLearningRate * step / warmupSteps; }
    switch (type)
    {
    case SchedulerType::Cosine_Decay :
        return minimumLearningRate
             + (0.5 * (initialLearningRate - minimumLearningRate)
                * (1 + std::cos(step * std::numbers::pi / (totalSteps - warmupSteps))));
    default :
        return initialLearningRate;
    }
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
void Scheduler::setParameters(double        initialLearningRate,
                              double        minimumLearningRate,
                              uint32_t      totalSteps,
                              SchedulerType schedulerType,
                              bool          enableWarmup)
#pragma clang diagnostic pop
{
    this->type                = schedulerType;
    this->initialLearningRate = initialLearningRate;
    this->minimumLearningRate = minimumLearningRate;
    this->totalSteps          = totalSteps;
    warmupEnabled             = enableWarmup;
    warmupSteps               = warmupEnabled * totalSteps * 0.05;
}