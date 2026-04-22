#include "ocp_constraint/position_constraint.h"
#include <ocp_solver/ocp_pre_computation.h>

namespace ocp_constraint {
  PositionConstraint::PositionConstraint(const pinocchio::FrameIndex targetFrameId,
                                         const Eigen::Vector3d& targetPosition)
    : ocs2::StateConstraint(ocs2::ConstraintOrder::Linear),
      targetFrameId_(targetFrameId),
      targetPosition_(targetPosition) {}

  PositionConstraint* PositionConstraint::clone() const {
    return new PositionConstraint(*this);
  }

  ocs2::vector_t PositionConstraint::getValue(ocs2::scalar_t time,
                                                 const ocs2::vector_t& state,
                                                 const ocs2::PreComputation& preComp) const {

    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const Eigen::Vector3d pos = pinocchioInterface.getData().oMf[targetFrameId_].translation();
    return pos - targetPosition_;
  }

  ocs2::VectorFunctionLinearApproximation PositionConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                     const ocs2::vector_t& state,
                                                                                     const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();

    const Eigen::Vector3d pos = pinocchioInterface.getData().oMf[targetFrameId_].translation();
    const Eigen::Vector3d error = pos - targetPosition_;

    Eigen::Matrix<double, 6, Eigen::Dynamic> J6(6, pinocchioInterface.getModel().nv);
    pinocchio::getFrameJacobian(pinocchioInterface.getModel(), pinocchioInterface.getData(), targetFrameId_, pinocchio::LOCAL_WORLD_ALIGNED, J6);
    Eigen::MatrixXd J = J6.topRows<3>();

    const size_t nx = state.size();
    const size_t nv = pinocchioInterface.getModel().nv;

    Eigen::MatrixXd dgdX = Eigen::MatrixXd::Zero(3, nx);
    dgdX.leftCols(nv) = J;

    ocs2::VectorFunctionLinearApproximation approx;
    approx.f = error;
    approx.dfdx = dgdX;

    return approx;
  }
}
