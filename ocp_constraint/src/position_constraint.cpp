#include "ocp_constraint/position_constraint.h"
#include <ocp_solver/ocp_pre_computation.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocp_constraint {
  PositionConstraint::PositionConstraint(const pinocchio::FrameIndex targetFrameId,
                                         const pinocchio::SE3& targetPose)
    : ocs2::StateConstraint(ocs2::ConstraintOrder::Linear),
      targetFrameId_(targetFrameId),
      targetPose_(targetPose) {}

  PositionConstraint* PositionConstraint::clone() const {
    return new PositionConstraint(*this);
  }

  bool PositionConstraint::isActive(ocs2::scalar_t time) const {
    return isActive_;
  }

  ocs2::vector_t PositionConstraint::getValue(ocs2::scalar_t time,
                                              const ocs2::vector_t& state,
                                              const ocs2::PreComputation& preComp) const {

    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    Eigen::VectorXd error = Eigen::VectorXd::Zero(6);
    error.head(3) = pinocchioInterface.getData().oMf[targetFrameId_].translation() - targetPose_.translation();
    error.tail(3) = ocs2::rotationErrorInLocal(pinocchioInterface.getData().oMf[targetFrameId_].rotation(), targetPose_.rotation());
    return error;
  }

  ocs2::VectorFunctionLinearApproximation PositionConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                     const ocs2::vector_t& state,
                                                                                     const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();

    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, pinocchioInterface.getModel().nv);
    pinocchio::getFrameJacobian(pinocchioInterface.getModel(), pinocchioInterface.getData(), targetFrameId_, pinocchio::LOCAL_WORLD_ALIGNED, J);

    const size_t nx = state.size();
    const size_t nv = pinocchioInterface.getModel().nv;

    Eigen::MatrixXd dgdX = Eigen::MatrixXd::Zero(6, nx);
    dgdX.leftCols(nv) = J;

    ocs2::VectorFunctionLinearApproximation approx;
    approx.f = getValue(time, state, preComp);
    approx.dfdx = dgdX;

    return approx;
  }
}
