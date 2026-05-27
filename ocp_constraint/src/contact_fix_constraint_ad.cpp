#include "ocp_constraint/contact_fix_constraint_ad.h"

namespace ocp_constraint {
  ContactFixConstraintAD::ContactFixConstraintAD(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                                 const ocp_solver::PinocchioFrameDynamicsCppAd& frameDynamics,
                                                 size_t numConstraints,
                                                 PositionConstraintAD::Config config)
    : PositionConstraintAD(frameDynamics, numConstraints, config),
      referenceManagerPtr_(&referenceManager) {}

  ContactFixConstraintAD::ContactFixConstraintAD(const ContactFixConstraintAD& rhs)
    : PositionConstraintAD(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_) {}

  bool ContactFixConstraintAD::isActive(ocs2::scalar_t time) const {
    return referenceManagerPtr_->isInContact(time, getFrameDynamics().getFrameIds()[0]);
  }

  const pinocchio::SE3 ContactFixConstraintAD::getTargetPose(ocs2::scalar_t time) const {
    pinocchio::SE3 targetPose = pinocchio::SE3::Identity();
    for (const std::pair<pinocchio::FrameIndex, pinocchio::SE3> contact : referenceManagerPtr_->getContacts(time)) {
      if (contact.first == getFrameDynamics().getFrameIds()[0]) targetPose = contact.second;
    }
    return targetPose;
  }

}
