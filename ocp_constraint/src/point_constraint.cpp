#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include "ocp_constraint/point_constraint.h"

namespace ocp_constraint {
  PointConstraint::PointConstraint(const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                                   const pinocchio::SE3 targetPose)
    : StateConstraint(ocs2::ConstraintOrder::Linear),
      targetPose_(targetPose),
      frameDynamicsPtr_(frameDynamics.clone()) {}

  PointConstraint::PointConstraint(const PointConstraint& rhs)
    : StateConstraint(rhs),
      frameDynamicsPtr_(rhs.frameDynamicsPtr_->clone()),
      targetPose_(rhs.targetPose_) {}

  ocs2::vector_t PointConstraint::getValue(ocs2::scalar_t time,
                                           const ocs2::vector_t& state,
                                           const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::vector_t constraint(6);
    constraint = frameDynamicsPtr_->getPosition(ocpPreComp) - targetPose_.translation(),
        frameDynamicsPtr_->getOrientationError(ocpPreComp, ocs2::matrixToQuaternion(targetPose_.rotation()));
    return constraint;
  }

  ocs2::VectorFunctionLinearApproximation PointConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                  const ocs2::vector_t& state,
                                                                                  const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);

    ocs2::VectorFunctionLinearApproximation approximation = ocs2::VectorFunctionLinearApproximation(6, 2*ocpPreComp.getPinocchioInterface().getModel().nv, 0);

    const ocs2::VectorFunctionLinearApproximation positionApprox = frameDynamicsPtr_->getPositionLinearApproximation(ocpPreComp);
    const ocs2::VectorFunctionLinearApproximation orientationApprox =
      frameDynamicsPtr_->getOrientationErrorLinearApproximation(ocpPreComp, ocs2::matrixToQuaternion(targetPose_.rotation()));

    approximation.f.head(3).noalias() += positionApprox.f - targetPose_.translation();
    approximation.f.tail(3).noalias() += orientationApprox.f;
    approximation.dfdx.topRows(3).noalias() += positionApprox.dfdx;
    approximation.dfdx.bottomRows(3).noalias() += orientationApprox.dfdx;

    return approximation;
  }

}
