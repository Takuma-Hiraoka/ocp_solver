#pragma once

#include <mutex>

#include <ocs2_core/misc/Lookup.h>
#include <ocs2_core/reference/ModeSchedule.h>
#include "ocp_solver/contact_candidate.h"
#include "ocp_solver/solver/trajectory.h"

namespace ocp_solver {
  using ContactTargetTrajectory = std::pair<ContactCandidateIndex, TargetSE3Trajectory>;

  struct ContactSchedule : ocs2::ModeSchedule {
  public:
    ContactSchedule() : ContactSchedule(std::vector<ocs2::scalar_t>{}, std::vector<std::vector<ContactTargetTrajectory> >{{{0, TargetSE3Trajectory(pinocchio::SE3::Identity())}}}) {}
    ContactSchedule(std::vector<ocs2::scalar_t> eventTimes_, std::vector<std::vector<ContactTargetTrajectory> > contactSequence_);

    std::vector<ContactTargetTrajectory> contactAtTime(ocs2::scalar_t time) const;

    void clean () {
      eventTimes.clear();
      modeSequence.clear();
      contactSequence.clear();
    }

    // std::vector<size_t> modeSequence;  // mode sequence of size N. not used
    std::vector<std::vector<ContactTargetTrajectory> > contactSequence; // [0] size N. corresponding to eventTimes. [1] contact vector. [2] candidate and target trajectory
  };
  ContactTargetTrajectory makeContactTarget(ContactCandidateIndex index, ocs2::scalar_t time, const pinocchio::SE3& pose, bool interpolate = false);
  void swap(ContactSchedule& lh, ContactSchedule& rh);

  std::ostream& operator<<(std::ostream& stream, const ContactSchedule& contactSchedule);

}
