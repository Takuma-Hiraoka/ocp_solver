#include "ocp_solver/contact_schedule.h"
#include <ocs2_core/misc/Display.h>

namespace ocp_solver {
  ContactSchedule::ContactSchedule(std::vector<ocs2::scalar_t> eventTimes_, std::vector<std::vector<std::pair<ContactCandidateIndex, pinocchio::SE3> > > contactSequence_)
    : ModeSchedule(eventTimes_, std::vector<size_t>(eventTimes_.size()+1, 0)), contactSequence(std::move(contactSequence_)) {
    assert(!contactSequence.empty());
    assert(eventTimes.size() + 1 == contactSequence.size());
  }

  std::vector<std::pair<ContactCandidateIndex, pinocchio::SE3> > ContactSchedule::contactAtTime(ocs2::scalar_t time) const {
    const size_t ind = ocs2::lookup::findIndexInTimeArray(eventTimes, time);
    return contactSequence[ind];
  }

  void swap(ContactSchedule& lh, ContactSchedule& rh) {
    lh.eventTimes.swap(rh.eventTimes);
    lh.contactSequence.swap(rh.contactSequence);
  }

  std::ostream& operator<<(std::ostream& stream, const ContactSchedule& contactSchedule) {
    stream << "event times:   {" << ocs2::toDelimitedString(contactSchedule.eventTimes) << "}\n";
    for (size_t i=0; i<contactSchedule.contactSequence.size(); i++) {
      for (size_t j=0; j<contactSchedule.contactSequence[i].size(); j++) {
        stream << "contact sequence: " << i << " " << j << "\n";
        stream << "candidate: contactIndex " << contactSchedule.contactSequence[i][j].first << "\n";
        stream << "target" << "\n";
        stream << "pos" << "\n";
        stream << contactSchedule.contactSequence[i][j].second.translation() << "\n";
        stream << "rot" << "\n";
        stream << contactSchedule.contactSequence[i][j].second.rotation() << "\n";
      }
    }
    return stream;
  }

}
