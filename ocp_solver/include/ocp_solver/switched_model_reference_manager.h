#pragma once

#include <ocs2_core/thread_support/Synchronized.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include "ocp_solver/contact_schedule.h"

namespace ocp_solver {
  class SwitchedModelReferenceManager : public ocs2::ReferenceManager {
  public:
    SwitchedModelReferenceManager(std::shared_ptr<ContactSchedule> contactSchedulePtr);

    ~SwitchedModelReferenceManager() override = default;

    void setModeSchedule(const ocs2::ModeSchedule& modeSchedule) override;

    size_t getContactFlags(ocs2::scalar_t time) const;

    const std::shared_ptr<ContactSchedule>& getContactSchedule() { return contactSchedulePtr_; }

  private:
    void modifyReferences(ocs2::scalar_t initTime, ocs2::scalar_t finalTime, const ocs2::vector_t& initState, ocs2::TargetTrajectories& targetTrajectories,
                          ocs2::ModeSchedule& modeSchedule) override;

    std::shared_ptr<ContactSchedule> contactSchedulePtr_;
  };
}
