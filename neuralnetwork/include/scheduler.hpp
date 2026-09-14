#pragma once

#include <cstdint>

enum class SchedulerType
{
    None,
    Cosine_Decay
};

class Scheduler
{
public:
    [[nodiscard]] double getLearningRate(uint32_t step) const;

    void setParameters(double        initialLearningRate,
                       double        minimumLearningRate,
                       uint32_t      totalSteps,
                       SchedulerType schedulerType,
                       bool          enableWarmup = false);

private:
    SchedulerType type;

    double   initialLearningRate;
    double   minimumLearningRate;
    uint32_t totalSteps;
    bool     warmupEnabled;
    uint32_t warmupSteps;
};