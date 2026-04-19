#pragma once

#include <mutex>

#include <ocs2_core/misc/Lookup.h>
#include <ocs2_core/reference/ModeSchedule.h>

namespace ocp_solver {
  class ContactSchedule {
  public:
    ContactSchedule(ocs2::ModeSchedule initModeSchedule, ocs2::scalar_t phaseTransitionIdleTime);

    void setModeSchedule(const ocs2::ModeSchedule& modeSchedule) { modeSchedule_ = modeSchedule; }

    ocs2::ModeSchedule getModeSchedule(ocs2::scalar_t lowerBoundTime, ocs2::scalar_t upperBoundTime);

  private:
    ocs2::ModeSchedule modeSchedule_;
    ocs2::scalar_t phaseTransitionIdleTime_;
  };
}
