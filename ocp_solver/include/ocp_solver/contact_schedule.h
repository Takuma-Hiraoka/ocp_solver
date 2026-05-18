#pragma once

#include <mutex>

#include <ocs2_core/misc/Lookup.h>
#include <ocs2_core/reference/ModeSchedule.h>
#include "ocp_solver/contact_candidate.h"

namespace ocp_solver {
  struct ContactSchedule : ocs2::ModeSchedule {
  public:
    ContactSchedule() : ContactSchedule(std::vector<ocs2::scalar_t>{}, std::vector<std::vector<std::pair<pinocchio::FrameIndex, pinocchio::SE3> > >{{{0, pinocchio::SE3::Identity()}}}) {}
    ContactSchedule(std::vector<ocs2::scalar_t> eventTimes_, std::vector<std::vector<std::pair<pinocchio::FrameIndex, pinocchio::SE3> > > contactSequence_);

    std::vector<std::pair<pinocchio::FrameIndex, pinocchio::SE3> > contactAtTime(ocs2::scalar_t time) const;

    void clean () {
      eventTimes.clear();
      modeSequence.clear();
      contactSequence.clear();
    }

    // std::vector<size_t> modeSequence;  // mode sequence of size N. not used
    std::vector<std::vector<std::pair<pinocchio::FrameIndex, pinocchio::SE3> > > contactSequence; // [0] size N. corresponding to eventTimes. [1] contact vector. [2] candidate and targetPose
  };
  void swap(ContactSchedule& lh, ContactSchedule& rh);

  std::ostream& operator<<(std::ostream& stream, const ContactSchedule& contactSchedule);

}
