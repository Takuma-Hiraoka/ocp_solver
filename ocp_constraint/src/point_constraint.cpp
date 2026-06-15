#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include "ocp_constraint/point_constraint.h"

#include <utility>

namespace ocp_constraint {
  PointConstraint::PointConstraint(const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                                   ocp_solver::TargetSE3Trajectory targetTrajectory)
    : StateConstraint(ocs2::ConstraintOrder::Linear),
      targetTrajectory_(std::move(targetTrajectory)),
      frameDynamicsPtr_(frameDynamics.clone()) {}

  PointConstraint::PointConstraint(const PointConstraint& rhs)
    : StateConstraint(rhs),
      frameDynamicsPtr_(rhs.frameDynamicsPtr_->clone()),
      targetTrajectory_(rhs.targetTrajectory_) {}

  ocs2::vector_t PointConstraint::getValue(ocs2::scalar_t time,
                                           const ocs2::vector_t& state,
                                           const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    const pinocchio::SE3 targetPose = targetTrajectory_.getTargetPose(time);
    ocs2::vector_t constraint(6);
    constraint << frameDynamicsPtr_->getPosition(ocpPreComp) - targetPose.translation(),
        frameDynamicsPtr_->getOrientationError(ocpPreComp, ocs2::matrixToQuaternion(targetPose.rotation()));
    return constraint;
  }

  ocs2::VectorFunctionLinearApproximation PointConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                  const ocs2::vector_t& state,
                                                                                  const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);

    const auto& model = ocpPreComp.getPinocchioInterface().getModel();
    const size_t stateVariableDim = state.size() - (model.nq - model.nv);
    ocs2::VectorFunctionLinearApproximation approximation =
      ocs2::VectorFunctionLinearApproximation::Zero(6, stateVariableDim, 0);

    const pinocchio::SE3 targetPose = targetTrajectory_.getTargetPose(time);
    const ocs2::VectorFunctionLinearApproximation positionApprox = frameDynamicsPtr_->getPositionLinearApproximation(ocpPreComp);
    const ocs2::VectorFunctionLinearApproximation orientationApprox =
      frameDynamicsPtr_->getOrientationErrorLinearApproximation(ocpPreComp, ocs2::matrixToQuaternion(targetPose.rotation()));

    approximation.f.head(3).noalias() += positionApprox.f - targetPose.translation();
    approximation.f.tail(3).noalias() += orientationApprox.f;
    approximation.dfdx.topRows(3).noalias() += positionApprox.dfdx;
    approximation.dfdx.bottomRows(3).noalias() += orientationApprox.dfdx;

    return approximation;
  }

}
