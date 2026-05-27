#include "ocp_constraint/contact_fix_constraint.h"

namespace ocp_constraint {
  ContactFixConstraint::ContactFixConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                             const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                                             size_t numConstraints,
                                             Config config)
    : PositionConstraint(frameDynamics, numConstraints, config),
      referenceManagerPtr_(&referenceManager) {}

  ContactFixConstraint::ContactFixConstraint(const ContactFixConstraint& rhs)
    : PositionConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_) {}

  bool ContactFixConstraint::isActive(ocs2::scalar_t time) const {
    return referenceManagerPtr_->isInContact(time, getFrameDynamics().getFrameId());
  }

  const pinocchio::SE3 ContactFixConstraint::getTargetPose(ocs2::scalar_t time) const {
    pinocchio::SE3 targetPose = pinocchio::SE3::Identity();
    for (const std::pair<pinocchio::FrameIndex, pinocchio::SE3> contact : referenceManagerPtr_->getContacts(time)) {
      if (contact.first == getFrameDynamics().getFrameId()) targetPose = contact.second;
    }
    return targetPose;
  }

}
