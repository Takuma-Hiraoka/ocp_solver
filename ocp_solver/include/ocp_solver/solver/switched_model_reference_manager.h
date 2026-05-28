#pragma once

#include <ocs2_core/thread_support/Synchronized.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include "ocp_solver/contact_schedule.h"

namespace ocp_solver {
  class SwitchedModelReferenceManager : public ocs2::ReferenceManager {
  public:
    SwitchedModelReferenceManager(ContactSchedule contactSchedule = ContactSchedule());

    ~SwitchedModelReferenceManager() override = default;

    const ContactSchedule& getContactSchedule() const { return contactSchedule_.get(); }
    void setContactSchedule(const ContactSchedule& contactSchedule) { contactSchedule_.setBuffer(contactSchedule); }
    void setContactSchedule(ContactSchedule&& contactSchedule) { contactSchedule_.setBuffer(std::move(contactSchedule)); }

    const std::vector<std::pair<ContactCandidateIndex, pinocchio::SE3> > getContacts(ocs2::scalar_t time) const;

    bool isInContact(ocs2::scalar_t time, ContactCandidateIndex index) const;

  private:
    void modifyReferences(ocs2::scalar_t initTime, ocs2::scalar_t finalTime, const ocs2::vector_t& initState, ocs2::TargetTrajectories& targetTrajectories,
                          ocs2::ModeSchedule& modeSchedule) override;

    ocs2::BufferedValue<ContactSchedule> contactSchedule_;
  };
}
