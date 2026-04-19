#include "ocp_solver/contact_schedule.h"

namespace ocp_solver {
  ContactSchedule::ContactSchedule(ocs2::ModeSchedule initModeSchedule, ocs2::scalar_t phaseTransitionIdleTime)
    : modeSchedule_(std::move(initModeSchedule)),
      phaseTransitionIdleTime_(phaseTransitionIdleTime) {}

  ocs2::ModeSchedule ContactSchedule::getModeSchedule(ocs2::scalar_t lowerBoundTime, ocs2::scalar_t upperBoundTime) {
    return modeSchedule_;
  }

}
