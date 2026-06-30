#include "ocp_solver/contact_schedule.h"
#include <ocs2_core/misc/Display.h>
#include <algorithm>
#include <stdexcept>

namespace ocp_solver {
  ContactSchedule::ContactSchedule(std::vector<ocs2::scalar_t> eventTimes_, std::vector<std::vector<ContactTargetTrajectory> > contactSequence_)
    : ModeSchedule(eventTimes_, std::vector<size_t>(eventTimes_.size()+1, 0)), contactSequence(std::move(contactSequence_)) {
    if (contactSequence.empty()) {
      throw std::runtime_error("ContactSchedule requires at least one contact segment.");
    }
    if (eventTimes.size() + 1 != contactSequence.size()) {
      throw std::runtime_error("ContactSchedule requires contactSequence.size() == eventTimes.size() + 1.");
    }
  }

  std::vector<ContactTargetTrajectory> ContactSchedule::contactAtTime(ocs2::scalar_t time) const {
    const size_t ind = ocs2::lookup::findIndexInTimeArray(eventTimes, time);
    if (ind >= contactSequence.size()) {
      return contactSequence.back();
    }
    return contactSequence[ind];
  }

  ContactTargetTrajectory makeContactTarget(ContactCandidateIndex index, ocs2::scalar_t time, const pinocchio::SE3& pose, bool interpolate) {
    return {index, TargetSE3Trajectory({time}, {pose}, {ocs2::vector_t::Zero(6)}, {ocs2::vector_t::Zero(6)}, interpolate)};
  }

  void swap(ContactSchedule& lh, ContactSchedule& rh) {
    lh.eventTimes.swap(rh.eventTimes);
    lh.modeSequence.swap(rh.modeSequence);
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
        stream << contactSchedule.contactSequence[i][j].second.getTargetPose(
          contactSchedule.eventTimes.empty() ? 0.0 : contactSchedule.eventTimes[std::min(i, contactSchedule.eventTimes.size() - 1)]).translation() << "\n";
        stream << "rot" << "\n";
        stream << contactSchedule.contactSequence[i][j].second.getTargetPose(
          contactSchedule.eventTimes.empty() ? 0.0 : contactSchedule.eventTimes[std::min(i, contactSchedule.eventTimes.size() - 1)]).rotation() << "\n";
      }
    }
    return stream;
  }

}
