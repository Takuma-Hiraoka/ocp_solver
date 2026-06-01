#include "ocp_constraint/contact_fix_constraint.h"
#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocp_constraint {
  namespace {
    Eigen::Matrix3d makeAntiparallelNormalTargetRotation(const Eigen::Matrix3d& targetRotation) {
      Eigen::Matrix3d rotation = targetRotation;
      rotation.col(1) *= -1.0;
      rotation.col(2) *= -1.0;
      return rotation;
    }
  }

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
    for (const std::pair<ocp_solver::ContactCandidateIndex, pinocchio::SE3> contact : referenceManagerPtr_->getContacts(time)) {
      if (contact.first == getFrameDynamics().getFrameId()) targetPose = contact.second;
    }
    return targetPose;
  }

  ocs2::vector_t ContactFixConstraint::getValue(ocs2::scalar_t time,
                                                const ocs2::vector_t& state,
                                                const ocs2::vector_t& input,
                                                const ocs2::PreComputation& preComp) const {
    if (!getFrameDynamics().usesSearchedContactPoint()) {
      return PositionConstraint::getValue(time, state, input, preComp);
    }

    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::vector_t f = ocs2::vector_t::Zero(getNumConstraints(time));
    if (config_.Ax.size() > 0) {
      const pinocchio::SE3 targetPose = getTargetPose(time);
      const Eigen::Matrix3d targetRotation =
        makeAntiparallelNormalTargetRotation(targetPose.rotation());
      Eigen::Matrix<ocs2::scalar_t, 6, 1> xError = Eigen::Matrix<ocs2::scalar_t, 6, 1>::Zero();
      xError.head<3>() = frameDynamicsPtr_->getSearchedContactPointPosition(ocpPreComp) - targetPose.translation();
      xError.tail<3>() = frameDynamicsPtr_->getOrientationError(ocpPreComp, ocs2::matrixToQuaternion(targetRotation));
      xError(5) = 0.0;
      f.noalias() += config_.Ax * xError;
    }
    if (config_.Av.size() > 0) {
      f.noalias() += config_.Av * (frameDynamicsPtr_->getTwist(ocpPreComp) - getTargetTwist(time));
    }
    if (config_.Aa.size() > 0) {
      f.noalias() += config_.Aa * (frameDynamicsPtr_->getAccelerations(ocpPreComp) - getTargetAcc(time));
    }
    return f;
  }

  ocs2::VectorFunctionLinearApproximation ContactFixConstraint::getLinearApproximation(
      ocs2::scalar_t time,
      const ocs2::vector_t& state,
      const ocs2::vector_t& input,
      const ocs2::PreComputation& preComp) const {
    if (!getFrameDynamics().usesSearchedContactPoint()) {
      return PositionConstraint::getLinearApproximation(time, state, input, preComp);
    }

    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const size_t stateVariableDim = state.size() - (pinocchioInterface.getModel().nq - pinocchioInterface.getModel().nv);
    ocs2::VectorFunctionLinearApproximation linearApproximation =
      ocs2::VectorFunctionLinearApproximation::Zero(getNumConstraints(time), stateVariableDim, input.size());

    if (config_.Ax.size() > 0) {
      const pinocchio::SE3 targetPose = getTargetPose(time);
      const Eigen::Matrix3d targetRotation =
        makeAntiparallelNormalTargetRotation(targetPose.rotation());
      const ocs2::VectorFunctionLinearApproximation positionApprox =
        frameDynamicsPtr_->getSearchedContactPointPositionLinearApproximation(ocpPreComp);
      const ocs2::VectorFunctionLinearApproximation orientationApprox =
        frameDynamicsPtr_->getOrientationErrorLinearApproximation(ocpPreComp, ocs2::matrixToQuaternion(targetRotation));

      ocs2::VectorFunctionLinearApproximation poseApprox =
        ocs2::VectorFunctionLinearApproximation::Zero(6, stateVariableDim, input.size());
      poseApprox.f.head<3>() = positionApprox.f - targetPose.translation();
      poseApprox.f.segment<2>(3) = orientationApprox.f.head<2>();
      poseApprox.dfdx.topRows<3>().noalias() = positionApprox.dfdx;
      poseApprox.dfdx.middleRows<2>(3).noalias() = orientationApprox.dfdx.topRows<2>();

      linearApproximation.f.noalias() += config_.Ax * poseApprox.f;
      linearApproximation.dfdx.noalias() += config_.Ax * poseApprox.dfdx;
    }

    if (config_.Av.size() > 0) {
      const ocs2::VectorFunctionLinearApproximation velocityApprox = frameDynamicsPtr_->getTwistLinearApproximation(ocpPreComp);
      linearApproximation.f.noalias() += config_.Av * (velocityApprox.f - getTargetTwist(time));
      linearApproximation.dfdx.noalias() += config_.Av * velocityApprox.dfdx;
      linearApproximation.dfdu.noalias() += config_.Av * velocityApprox.dfdu;
    }

    if (config_.Aa.size() > 0) {
      const ocs2::VectorFunctionLinearApproximation accelApprox = frameDynamicsPtr_->getAccelerationsLinearApproximation(ocpPreComp);
      linearApproximation.f.noalias() += config_.Aa * (accelApprox.f - getTargetAcc(time));
      linearApproximation.dfdx.noalias() += config_.Aa * accelApprox.dfdx;
      linearApproximation.dfdu.noalias() += config_.Aa * accelApprox.dfdu;
    }

    return linearApproximation;
  }

}
