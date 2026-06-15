#include "ocp_solver/solver/switched_model_reference_manager.h"

namespace ocp_solver {
  SwitchedModelReferenceManager::SwitchedModelReferenceManager(ContactSchedule contactSchedule)
    : ReferenceManager(ocs2::TargetTrajectories(), ocs2::ModeSchedule()),
      contactSchedule_(std::move(contactSchedule)) {}

  std::vector<ContactTargetTrajectory> SwitchedModelReferenceManager::getContacts(ocs2::scalar_t time) const {
    return this->getContactSchedule().contactAtTime(time);
  }

  bool SwitchedModelReferenceManager::isInContact(ocs2::scalar_t time, ContactCandidateIndex index) const {
    const std::vector<ContactTargetTrajectory> contacts = this->getContacts(time);
    for (const ContactTargetTrajectory& contact : contacts) {
      if (contact.first == index) return true;
    }
    return false;
  }

  void SwitchedModelReferenceManager::modifyReferences(ocs2::scalar_t initTime, ocs2::scalar_t finalTime, const ocs2::vector_t& initState,
                                                       ocs2::TargetTrajectories& targetTrajectories, ocs2::ModeSchedule& modeSchedule) {
    contactSchedule_.updateFromBuffer();
    const ocs2::scalar_t timeHorizon = finalTime - initTime;
  }
}
