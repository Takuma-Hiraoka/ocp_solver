#include "ocp_solver/solver/switched_model_reference_manager.h"

namespace ocp_solver {
  namespace {
    bool isApprox(const ocs2::vector_t& lhs, const ocs2::vector_t& rhs) {
      return lhs.size() == rhs.size() && lhs.isApprox(rhs);
    }

    bool isApprox(const pinocchio::SE3& lhs, const pinocchio::SE3& rhs) {
      return lhs.translation().isApprox(rhs.translation())
             && lhs.rotation().isApprox(rhs.rotation());
    }

    void extendTargetTrajectoriesToHorizon(ocs2::TargetTrajectories& targetTrajectories,
                                           ocs2::scalar_t initTime,
                                           ocs2::scalar_t finalTime,
                                           const ocs2::vector_t& initState) {
      if (targetTrajectories.empty()) {
        targetTrajectories.timeTrajectory.push_back(initTime);
        targetTrajectories.stateTrajectory.push_back(initState);
        if (finalTime > initTime) {
          targetTrajectories.timeTrajectory.push_back(finalTime);
          targetTrajectories.stateTrajectory.push_back(initState);
        }
        return;
      }

      if (initTime < targetTrajectories.timeTrajectory.front()) {
        targetTrajectories.timeTrajectory.insert(targetTrajectories.timeTrajectory.begin(), initTime);
        targetTrajectories.stateTrajectory.insert(targetTrajectories.stateTrajectory.begin(),
                                                  targetTrajectories.stateTrajectory.front());
        if (!targetTrajectories.inputTrajectory.empty()) {
          targetTrajectories.inputTrajectory.insert(targetTrajectories.inputTrajectory.begin(),
                                                    targetTrajectories.inputTrajectory.front());
        }
      }

      if (finalTime > targetTrajectories.timeTrajectory.back()) {
        const bool hasHeldFinalState =
          targetTrajectories.stateTrajectory.size() >= 2
          && isApprox(targetTrajectories.stateTrajectory.back(),
                      targetTrajectories.stateTrajectory[targetTrajectories.stateTrajectory.size() - 2]);
        const bool hasHeldFinalInput =
          targetTrajectories.inputTrajectory.empty()
          || (targetTrajectories.inputTrajectory.size() >= 2
              && isApprox(targetTrajectories.inputTrajectory.back(),
                          targetTrajectories.inputTrajectory[targetTrajectories.inputTrajectory.size() - 2]));

        if (hasHeldFinalState && hasHeldFinalInput) {
          targetTrajectories.timeTrajectory.back() = finalTime;
        } else {
          targetTrajectories.timeTrajectory.push_back(finalTime);
          targetTrajectories.stateTrajectory.push_back(targetTrajectories.stateTrajectory.back());
          if (!targetTrajectories.inputTrajectory.empty()) {
            targetTrajectories.inputTrajectory.push_back(targetTrajectories.inputTrajectory.back());
          }
        }
      }
    }

    void extendTargetSE3TrajectoryToHorizon(TargetSE3Trajectory& targetTrajectory,
                                            ocs2::scalar_t initTime,
                                            ocs2::scalar_t finalTime) {
      if (targetTrajectory.timeTrajectory.empty()) {
        targetTrajectory = TargetSE3Trajectory();
      }

      if (initTime < targetTrajectory.timeTrajectory.front()) {
        targetTrajectory.timeTrajectory.insert(targetTrajectory.timeTrajectory.begin(), initTime);
        targetTrajectory.poseTrajectory.insert(targetTrajectory.poseTrajectory.begin(),
                                               targetTrajectory.poseTrajectory.front());
        targetTrajectory.twistTrajectory.insert(targetTrajectory.twistTrajectory.begin(),
                                                targetTrajectory.twistTrajectory.front());
        targetTrajectory.accTrajectory.insert(targetTrajectory.accTrajectory.begin(),
                                              targetTrajectory.accTrajectory.front());
      }

      if (finalTime > targetTrajectory.timeTrajectory.back()) {
        const bool hasHeldFinalPose =
          targetTrajectory.poseTrajectory.size() >= 2
          && isApprox(targetTrajectory.poseTrajectory.back(),
                      targetTrajectory.poseTrajectory[targetTrajectory.poseTrajectory.size() - 2]);
        const bool hasHeldFinalTwist =
          targetTrajectory.twistTrajectory.size() >= 2
          && isApprox(targetTrajectory.twistTrajectory.back(),
                      targetTrajectory.twistTrajectory[targetTrajectory.twistTrajectory.size() - 2]);
        const bool hasHeldFinalAcc =
          targetTrajectory.accTrajectory.size() >= 2
          && isApprox(targetTrajectory.accTrajectory.back(),
                      targetTrajectory.accTrajectory[targetTrajectory.accTrajectory.size() - 2]);

        if (hasHeldFinalPose && hasHeldFinalTwist && hasHeldFinalAcc) {
          targetTrajectory.timeTrajectory.back() = finalTime;
        } else {
          targetTrajectory.timeTrajectory.push_back(finalTime);
          targetTrajectory.poseTrajectory.push_back(targetTrajectory.poseTrajectory.back());
          targetTrajectory.twistTrajectory.push_back(targetTrajectory.twistTrajectory.back());
          targetTrajectory.accTrajectory.push_back(targetTrajectory.accTrajectory.back());
        }
      }
    }

    void extendContactScheduleTargetsToHorizon(ContactSchedule& contactSchedule,
                                               ocs2::scalar_t initTime,
                                               ocs2::scalar_t finalTime) {
      for (std::vector<ContactTargetTrajectory>& contacts : contactSchedule.contactSequence) {
        for (ContactTargetTrajectory& contact : contacts) {
          extendTargetSE3TrajectoryToHorizon(contact.second, initTime, finalTime);
        }
      }
    }

    void setModeScheduleFromContactSchedule(const ContactSchedule& contactSchedule,
                                            ocs2::ModeSchedule& modeSchedule) {
      modeSchedule.eventTimes = contactSchedule.eventTimes;
      if (contactSchedule.modeSequence.size() == contactSchedule.eventTimes.size() + 1) {
        modeSchedule.modeSequence = contactSchedule.modeSequence;
      } else {
        modeSchedule.modeSequence.assign(contactSchedule.eventTimes.size() + 1, 0);
      }
      if (modeSchedule.modeSequence.empty()) {
        modeSchedule.modeSequence.push_back(0);
      }
    }
  }  // namespace

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
    extendTargetTrajectoriesToHorizon(targetTrajectories, initTime, finalTime, initState);
    extendContactScheduleTargetsToHorizon(contactSchedule_.get(), initTime, finalTime);
    setModeScheduleFromContactSchedule(contactSchedule_.get(), modeSchedule);
  }
}
