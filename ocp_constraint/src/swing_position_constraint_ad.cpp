#include "ocp_constraint/swing_position_constraint_ad.h"

#include <algorithm>

#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocp_constraint {
  SwingPositionConstraintAD::SwingPositionConstraintAD(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                                       const ocp_solver::PinocchioFrameDynamicsCppAd& frameDynamics,
                                                       Config config,
                                                       ocs2::scalar_t ignoreTime,
                                                       double height,
                                                       double swingWeight)
    : StateConstraint(ocs2::ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      frameDynamicsPtr_(frameDynamics.clone()),
      config_(std::move(config)),
      ignoreTime_(ignoreTime),
      height_(height),
      swingWeight_(swingWeight){}

  SwingPositionConstraintAD::SwingPositionConstraintAD(const SwingPositionConstraintAD& rhs)
    : StateConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      frameDynamicsPtr_(rhs.frameDynamicsPtr_->clone()),
      config_(rhs.config_),
      ignoreTime_(rhs.ignoreTime_),
      height_(rhs.height_),
      swingWeight_(rhs.swingWeight_) {}

  void SwingPositionConstraintAD::configure(Config&& config) {
    assert((config.Ax.size() > 0 && config.Ax.rows() == numConstraints_) || config.Ax.size() == 0);
    assert((config.Ax.size() > 0 && config.Ax.cols() == 3) || config.Ax.size() == 0);
    config_ = std::move(config);
  }

  bool SwingPositionConstraintAD::isActive(ocs2::scalar_t time) const {
    bool inContact = referenceManagerPtr_->isInContact(time, frameDynamicsPtr_->getFrameIds()[0]);
    if (inContact) return false;
    std::vector<std::pair<ocs2::scalar_t, pinocchio::SE3> > nearestContacts = nearestContact(time);
    if ((nearestContacts[0].first != -1.0) && (nearestContacts[1].first != -1.0)) return true;
    if (((nearestContacts[0].first != -1.0) && (time - nearestContacts[0].first) < ignoreTime_) ||
        ((nearestContacts[1].first != -1.0) && (nearestContacts[1].first - time) < ignoreTime_)) return true;
    return false;
  }

  std::vector<std::pair<ocs2::scalar_t, pinocchio::SE3> > SwingPositionConstraintAD::nearestContact(ocs2::scalar_t time) const {
    ocp_solver::ContactSchedule contactSchedule = referenceManagerPtr_->getContactSchedule();
    ocs2::scalar_t beforeTime = -1.0;
    pinocchio::SE3 beforeSE3 = pinocchio::SE3::Identity();
    ocs2::scalar_t afterTime = -1.0;
    pinocchio::SE3 afterSE3 = pinocchio::SE3::Identity();
    for (size_t i=0; (i<contactSchedule.contactSequence.size()) && (afterTime == -1.0); i++) {
      for (const ocp_solver::ContactTargetTrajectory& contact : contactSchedule.contactSequence[i]) {
        if (contact.first == frameDynamicsPtr_->getFrameIds()[0]) {
          if ((i < contactSchedule.eventTimes.size()) && (contactSchedule.eventTimes[i] <= time)) {
            beforeTime = contactSchedule.eventTimes[i];
            beforeSE3 = contact.second.getTargetPose(beforeTime);
          }
          if ((i != 0) && (contactSchedule.eventTimes[i-1] > time)) {
            afterTime = contactSchedule.eventTimes[i-1];
            afterSE3 = contact.second.getTargetPose(afterTime);
          }
        }
      }
    }
    return {{beforeTime, beforeSE3}, {afterTime, afterSE3}};
  }

  ocs2::vector_t SwingPositionConstraintAD::getValue(ocs2::scalar_t time,
                                                     const ocs2::vector_t& state,
                                                     const ocs2::PreComputation& preComp) const {
    ocs2::vector_t f = ocs2::vector_t::Zero(3);
    if (config_.Ax.size() > 0) {
      pinocchio::SE3 targetPose = pinocchio::SE3::Identity();
      std::vector<std::pair<ocs2::scalar_t, pinocchio::SE3> > nearestContacts = nearestContact(time);
      if ((nearestContacts[0].first != -1.0) && (nearestContacts[1].first != -1.0)) {
        double ratio = (time - nearestContacts[0].first) / (nearestContacts[1].first - nearestContacts[0].first);
        if (ratio < (1.0 / (1.0+swingWeight_+1.0))) { // liftup
          double liftRatio = ratio / (1.0 / (1.0+swingWeight_+1.0));
          targetPose = nearestContacts[0].second;
          targetPose.translation() += targetPose.rotation() * Eigen::Vector3d(0.0, 0.0, height_ * liftRatio);
        } else if (ratio < ((1.0+swingWeight_) / (1.0+swingWeight_+1.0))) { // swing
          double swingRatio = (ratio - (1.0 / (1.0+swingWeight_+1.0))) / (swingWeight_ / (1.0+swingWeight_+1.0));
          targetPose.translation() = (nearestContacts[0].second.translation() + nearestContacts[0].second.rotation() * Eigen::Vector3d(0.0, 0.0, height_)) * (1 - swingRatio) + (nearestContacts[1].second.translation() + nearestContacts[1].second.rotation() * Eigen::Vector3d(0.0, 0.0, height_)) * swingRatio;
        } else { // touch down
          double downRatio = (ratio - ((1.0+swingWeight_) / (1.0+swingWeight_+1.0))) / (1.0 / (1.0+swingWeight_+1.0));
          targetPose = nearestContacts[1].second;
          targetPose.translation() += targetPose.rotation() * Eigen::Vector3d(0.0, 0.0, height_ * (1-downRatio));
        }
      } else if (nearestContacts[0].first != -1.0) { // lift
        double liftRatio = std::clamp((time - nearestContacts[0].first) / ignoreTime_, 0.0, 1.0);
        targetPose = nearestContacts[0].second;
        targetPose.translation() += targetPose.rotation() * Eigen::Vector3d(0.0, 0.0, height_ * liftRatio);
      } else if (nearestContacts[1].first != -1.0) { // down
        double downRatio = std::clamp((nearestContacts[1].first - time) / ignoreTime_, 0.0, 1.0);
        targetPose = nearestContacts[1].second;
        targetPose.translation() += targetPose.rotation() * Eigen::Vector3d(0.0, 0.0, height_ * (1.0 - downRatio));
      }
      // foot pose is a 6D vector containing the foot position and orientation error wrt. to the ground normal
      Eigen::Matrix<ocs2::scalar_t, 3, 1> xError;
      xError << frameDynamicsPtr_->getPosition(state).front() - targetPose.translation();
      f.noalias() += config_.Ax * xError;
    }
    return f;
  }

  ocs2::VectorFunctionLinearApproximation SwingPositionConstraintAD::getLinearApproximation(ocs2::scalar_t time,
                                                                                            const ocs2::vector_t& state,
                                                                                            const ocs2::PreComputation& preComp) const {
    ocs2::VectorFunctionLinearApproximation linearApproximation =
      ocs2::VectorFunctionLinearApproximation::Zero(3, state.size(), 0);

    // Orientation error gains are ignored for now
    // This is equal with assuming that the bottom 3 rows of Ax are zero.
    if (config_.Ax.size() > 0) {
      pinocchio::SE3 targetPose = pinocchio::SE3::Identity();
      std::vector<std::pair<ocs2::scalar_t, pinocchio::SE3> > nearestContacts = nearestContact(time);
      if ((nearestContacts[0].first != -1.0) && (nearestContacts[1].first != -1.0)) {
        double ratio = (time - nearestContacts[0].first) / (nearestContacts[1].first - nearestContacts[0].first);
        if (ratio < (1.0 / (1.0+swingWeight_+1.0))) { // liftup
          double liftRatio = ratio / (1.0 / (1.0+swingWeight_+1.0));
          targetPose = nearestContacts[0].second;
          targetPose.translation() += targetPose.rotation() * Eigen::Vector3d(0.0, 0.0, height_ * liftRatio);
        } else if (ratio < ((1.0+swingWeight_) / (1.0+swingWeight_+1.0))) { // swing
          double swingRatio = (ratio - (1.0 / (1.0+swingWeight_+1.0))) / (swingWeight_ / (1.0+swingWeight_+1.0));
          targetPose.translation() = (nearestContacts[0].second.translation() + nearestContacts[0].second.rotation() * Eigen::Vector3d(0.0, 0.0, height_)) * (1 - swingRatio) + (nearestContacts[1].second.translation() + nearestContacts[1].second.rotation() * Eigen::Vector3d(0.0, 0.0, height_)) * swingRatio;
        } else { // touch down
          double downRatio = (ratio - ((1.0+swingWeight_) / (1.0+swingWeight_+1.0))) / (1.0 / (1.0+swingWeight_+1.0));
          targetPose = nearestContacts[1].second;
          targetPose.translation() += targetPose.rotation() * Eigen::Vector3d(0.0, 0.0, height_ * (1-downRatio));
        }
      } else if (nearestContacts[0].first != -1.0) { // lift
        double liftRatio = std::clamp((time - nearestContacts[0].first) / ignoreTime_, 0.0, 1.0);
        targetPose = nearestContacts[0].second;
        targetPose.translation() += targetPose.rotation() * Eigen::Vector3d(0.0, 0.0, height_ * liftRatio);
      } else if (nearestContacts[1].first != -1.0) { // down
        double downRatio = std::clamp((nearestContacts[1].first - time) / ignoreTime_, 0.0, 1.0);
        targetPose = nearestContacts[1].second;
        targetPose.translation() += targetPose.rotation() * Eigen::Vector3d(0.0, 0.0, height_ * (1.0 - downRatio));
      }
      const ocs2::VectorFunctionLinearApproximation positionApprox = frameDynamicsPtr_->getPositionLinearApproximation(state).front();

      linearApproximation.f.head(3).noalias() += config_.Ax.topLeftCorner(3, 3) * (positionApprox.f - targetPose.translation());
      linearApproximation.dfdx.topRows(3).noalias() += config_.Ax.topLeftCorner(3, 3) * positionApprox.dfdx;
    }

    return linearApproximation;
  }

}
