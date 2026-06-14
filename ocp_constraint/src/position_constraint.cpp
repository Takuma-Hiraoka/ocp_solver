#include "ocp_constraint/position_constraint.h"
#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocp_constraint {
  PositionConstraint::PositionConstraint(const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                                         size_t numConstraints,
                                         Config config,
                                         pinocchio::SE3 targetPose,
                                         ocs2::vector_t targetTwist,
                                         ocs2::vector_t targetAcc)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      frameDynamicsPtr_(frameDynamics.clone()),
      numConstraints_(numConstraints),
      config_(std::move(config)),
      targetPose_(std::move(targetPose)),
      targetTwist_(std::move(targetTwist)),
      targetAcc_(std::move(targetAcc)) {}

  PositionConstraint::PositionConstraint(const PositionConstraint& rhs)
    : StateInputConstraint(rhs),
      frameDynamicsPtr_(rhs.frameDynamicsPtr_->clone()),
      numConstraints_(rhs.numConstraints_),
      config_(rhs.config_),
      targetPose_(rhs.targetPose_),
      targetTwist_(rhs.targetTwist_),
      targetAcc_(rhs.targetAcc_) {}

  void PositionConstraint::configure(Config&& config) {
    assert(config.Ax.size() > 0 || config.Av.size() > 0);
    assert((config.Ax.size() > 0 && config.Ax.cols() == 6) || config.Ax.size() == 0);
    assert((config.Av.size() > 0 && config.Av.cols() == 6) || config.Av.size() == 0);
    assert((config.Aa.size() > 0 && config.Aa.cols() == 6) || config.Aa.size() == 0);
    config_ = std::move(config);
  }

  size_t PositionConstraint::getConfiguredNumConstraints() const {
    size_t numConstraints = 0;
    if (config_.Ax.size() > 0) numConstraints += config_.Ax.rows();
    if (config_.Av.size() > 0) numConstraints += config_.Av.rows();
    if (config_.Aa.size() > 0) numConstraints += config_.Aa.rows();
    return numConstraints > 0 ? numConstraints : numConstraints_;
  }

  ocs2::vector_t PositionConstraint::getValue(ocs2::scalar_t time,
                                              const ocs2::vector_t& state,
                                              const ocs2::vector_t& input,
                                              const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::vector_t f = ocs2::vector_t::Zero(getNumConstraints(time));
    Eigen::Index row = 0;
    if (config_.Ax.size() > 0) {
      const pinocchio::SE3 targetPose = getTargetPose(time);
      Eigen::Matrix<ocs2::scalar_t, 6, 1> xError;
      xError << frameDynamicsPtr_->getPosition(ocpPreComp) - targetPose.translation(),
        frameDynamicsPtr_->getOrientationError(ocpPreComp, ocs2::matrixToQuaternion(targetPose.rotation()));
      f.segment(row, config_.Ax.rows()).noalias() = config_.Ax * xError;
      row += config_.Ax.rows();
    }
    if (config_.Av.size() > 0) {
      f.segment(row, config_.Av.rows()).noalias() =
        config_.Av * (frameDynamicsPtr_->getTwist(ocpPreComp) - getTargetTwist(time));
      row += config_.Av.rows();
    }
    if (config_.Aa.size() > 0) {
      f.segment(row, config_.Aa.rows()).noalias() =
        config_.Aa * (frameDynamicsPtr_->getAccelerations(ocpPreComp) - getTargetAcc(time));
    }
    return f;
  }

  ocs2::VectorFunctionLinearApproximation PositionConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                     const ocs2::vector_t& state,
                                                                                     const ocs2::vector_t& input,
                                                                                     const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const size_t stateVariableDim = state.size() - (pinocchioInterface.getModel().nq - pinocchioInterface.getModel().nv);
    ocs2::VectorFunctionLinearApproximation linearApproximation =
      ocs2::VectorFunctionLinearApproximation::Zero(getNumConstraints(time), stateVariableDim, input.size());

    Eigen::Index row = 0;

    if (config_.Ax.size() > 0) {
      const pinocchio::SE3 targetPose = getTargetPose(time);
      const ocs2::VectorFunctionLinearApproximation positionApprox = frameDynamicsPtr_->getPositionLinearApproximation(ocpPreComp);
      const ocs2::VectorFunctionLinearApproximation orientationApprox =
        frameDynamicsPtr_->getOrientationErrorLinearApproximation(ocpPreComp, ocs2::matrixToQuaternion(targetPose.rotation()));

      ocs2::VectorFunctionLinearApproximation poseApprox =
        ocs2::VectorFunctionLinearApproximation::Zero(6, stateVariableDim, input.size());
      poseApprox.f.head<3>() = positionApprox.f - targetPose.translation();
      poseApprox.f.tail<3>() = orientationApprox.f;
      poseApprox.dfdx.topRows<3>() = positionApprox.dfdx;
      poseApprox.dfdx.bottomRows<3>() = orientationApprox.dfdx;

      linearApproximation.f.segment(row, config_.Ax.rows()).noalias() = config_.Ax * poseApprox.f;
      linearApproximation.dfdx.middleRows(row, config_.Ax.rows()).noalias() = config_.Ax * poseApprox.dfdx;
      linearApproximation.dfdu.middleRows(row, config_.Ax.rows()).noalias() = config_.Ax * poseApprox.dfdu;
      row += config_.Ax.rows();
    }

    if (config_.Av.size() > 0) {
      const ocs2::VectorFunctionLinearApproximation velocityApprox = frameDynamicsPtr_->getTwistLinearApproximation(ocpPreComp);
      linearApproximation.f.segment(row, config_.Av.rows()).noalias() =
        config_.Av * (velocityApprox.f - getTargetTwist(time));
      linearApproximation.dfdx.middleRows(row, config_.Av.rows()).noalias() = config_.Av * velocityApprox.dfdx;
      linearApproximation.dfdu.middleRows(row, config_.Av.rows()).noalias() = config_.Av * velocityApprox.dfdu;
      row += config_.Av.rows();
    }

    if (config_.Aa.size() > 0) {
      const ocs2::VectorFunctionLinearApproximation accelApprox = frameDynamicsPtr_->getAccelerationsLinearApproximation(ocpPreComp);
      linearApproximation.f.segment(row, config_.Aa.rows()).noalias() =
        config_.Aa * (accelApprox.f - getTargetAcc(time));
      linearApproximation.dfdx.middleRows(row, config_.Aa.rows()).noalias() = config_.Aa * accelApprox.dfdx;
      linearApproximation.dfdu.middleRows(row, config_.Aa.rows()).noalias() = config_.Aa * accelApprox.dfdu;
    }

    return linearApproximation;
  }

}
