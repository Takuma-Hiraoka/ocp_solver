#include <ocp_solver/ocp_pre_computation.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include "ocp_constraint/point_constraint.h"

namespace ocp_constraint {
  PointConstraint::PointConstraint(const ocs2::PinocchioInterface pinocchioInterface,
                                   const ocp_solver::OCPPinocchioMapping mapping,
                                   const std::string targetFrameName,
                                   const pinocchio::SE3 targetPose)
    : StateConstraint(ocs2::ConstraintOrder::Linear),
      targetPose_(targetPose),
      pinocchioInterface_(pinocchioInterface),
      mapping_(mapping),
      targetFrameName_(targetFrameName) {
    endEffectorKinematicsPtr_.reset(new ocs2::PinocchioEndEffectorKinematics(pinocchioInterface, mapping, {targetFrameName}));
  }

  ocs2::vector_t PointConstraint::getValue(ocs2::scalar_t time,
                                           const ocs2::vector_t& state,
                                           const ocs2::PreComputation& preComp) const {
    if (endEffectorKinematicsPtr_ != nullptr) {
      const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
      endEffectorKinematicsPtr_->setPinocchioInterface(ocpPreComp.getPinocchioInterface());
    }
    ocs2::vector_t constraint(6);
    constraint.head<3>() = endEffectorKinematicsPtr_->getPosition(state).front() - targetPose_.translation();
    constraint.tail<3>() = endEffectorKinematicsPtr_->getOrientationError(state, {ocs2::matrixToQuaternion(targetPose_.rotation())}).front();
    return constraint;
  }

  ocs2::VectorFunctionLinearApproximation PointConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                  const ocs2::vector_t& state,
                                                                                  const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    if (endEffectorKinematicsPtr_ != nullptr) {
      endEffectorKinematicsPtr_->setPinocchioInterface(ocpPreComp.getPinocchioInterface());
    }

    ocs2::VectorFunctionLinearApproximation approximation = ocs2::VectorFunctionLinearApproximation(6, 2*ocpPreComp.getPinocchioInterface().getModel().nv, 0);

    const ocs2::VectorFunctionLinearApproximation  eePosition = endEffectorKinematicsPtr_->getPositionLinearApproximation(state).front();
    approximation.f.head<3>() = eePosition.f - targetPose_.translation();
    approximation.dfdx.topRows<3>() = eePosition.dfdx;

    const ocs2::VectorFunctionLinearApproximation  eeOrientationError = endEffectorKinematicsPtr_->getOrientationErrorLinearApproximation(state, {ocs2::matrixToQuaternion(targetPose_.rotation())}).front();
    approximation.f.tail<3>() = eeOrientationError.f;
    approximation.dfdx.bottomRows<3>() = eeOrientationError.dfdx;

    const ocs2::VectorFunctionLinearApproximation eeVelocity = endEffectorKinematicsPtr_->getVelocityLinearApproximation(state, state).front();

    return approximation;
  }

}
