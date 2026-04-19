#include "ocp_solver/switched_model_reference_manager.h"

namespace ocp_solver {
  SwitchedModelReferenceManager::SwitchedModelReferenceManager(std::shared_ptr<ContactSchedule> contactSchedulePtr)
    : ReferenceManager(ocs2::TargetTrajectories(), ocs2::ModeSchedule()),
      contactSchedulePtr_(std::move(contactSchedulePtr)) {}

  void SwitchedModelReferenceManager::setModeSchedule(const ocs2::ModeSchedule& modeSchedule) {
    ReferenceManager::setModeSchedule(modeSchedule);
    contactSchedulePtr_->setModeSchedule(modeSchedule);
  }

  size_t SwitchedModelReferenceManager::getContactFlags(ocs2::scalar_t time) const {
    return this->getModeSchedule().modeAtTime(time);
  }

  void SwitchedModelReferenceManager::modifyReferences(ocs2::scalar_t initTime, ocs2::scalar_t finalTime, const ocs2::vector_t& initState,
                                                       ocs2::TargetTrajectories& targetTrajectories, ocs2::ModeSchedule& modeSchedule) {
    const auto timeHorizon = finalTime - initTime;
    modeSchedule = contactSchedulePtr_->getModeSchedule(initTime - timeHorizon, finalTime + timeHorizon);
  }
}
