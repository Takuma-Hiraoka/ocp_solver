#include "ocp_constraint/position_constraint_ad.h"
#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocp_constraint {
  PositionConstraintAD::PositionConstraintAD(const ocp_solver::PinocchioFrameDynamicsCppAd& frameDynamics,
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

  PositionConstraintAD::PositionConstraintAD(const PositionConstraintAD& rhs)
    : StateInputConstraint(rhs),
      frameDynamicsPtr_(rhs.frameDynamicsPtr_->clone()),
      numConstraints_(rhs.numConstraints_),
      config_(rhs.config_),
      targetPose_(rhs.targetPose_),
      targetTwist_(rhs.targetTwist_),
      targetAcc_(rhs.targetAcc_) {}

  void PositionConstraintAD::configure(Config&& config) {
    assert(config.Ax.size() > 0 || config.Av.size() > 0);
    assert((config.Ax.size() > 0 && config.Ax.cols() == 6) || config.Ax.size() == 0);
    assert((config.Av.size() > 0 && config.Av.cols() == 6) || config.Av.size() == 0);
    assert((config.Aa.size() > 0 && config.Aa.cols() == 6) || config.Aa.size() == 0);
    config_ = std::move(config);
  }

  size_t PositionConstraintAD::getConfiguredNumConstraints() const {
    size_t numConstraints = 0;
    if (config_.Ax.size() > 0) numConstraints += config_.Ax.rows();
    if (config_.Av.size() > 0) numConstraints += config_.Av.rows();
    if (config_.Aa.size() > 0) numConstraints += config_.Aa.rows();
    return numConstraints > 0 ? numConstraints : numConstraints_;
  }

  ocs2::vector_t PositionConstraintAD::getValue(ocs2::scalar_t time,
                                                const ocs2::vector_t& state,
                                                const ocs2::vector_t& input,
                                                const ocs2::PreComputation& preComp) const {
    ocs2::vector_t f = ocs2::vector_t::Zero(getNumConstraints(time));
    Eigen::Index row = 0;
    if (config_.Ax.size() > 0) {
      const pinocchio::SE3 targetPose = getTargetPose(time);
      // foot pose is a 6D vector containing the foot position and orientation error wrt. to the ground normal
      Eigen::Matrix<ocs2::scalar_t, 6, 1> xError;
      xError << frameDynamicsPtr_->getPosition(state).front() - targetPose.translation(),
        frameDynamicsPtr_->getOrientationError(state, {ocs2::matrixToQuaternion(targetPose.rotation())}).front();
      f.segment(row, config_.Ax.rows()).noalias() = config_.Ax * xError;
      row += config_.Ax.rows();
    }
    if (config_.Av.size() > 0) {
      f.segment(row, config_.Av.rows()).noalias() =
        config_.Av * (frameDynamicsPtr_->getTwist(state, input).front() - getTargetTwist(time));
      row += config_.Av.rows();
    }
    if (config_.Aa.size() > 0) {
      f.segment(row, config_.Aa.rows()).noalias() =
        config_.Aa * (frameDynamicsPtr_->getAccelerations(state, input).front() - getTargetAcc(time));
    }
    return f;
  }

  ocs2::VectorFunctionLinearApproximation PositionConstraintAD::getLinearApproximation(ocs2::scalar_t time,
                                                                                       const ocs2::vector_t& state,
                                                                                       const ocs2::vector_t& input,
                                                                                       const ocs2::PreComputation& preComp) const {
    ocs2::VectorFunctionLinearApproximation linearApproximation =
      ocs2::VectorFunctionLinearApproximation::Zero(getNumConstraints(time), state.size(), input.size());

    Eigen::Index row = 0;
    // Orientation error gains are ignored for now
    // This is equal with assuming that the bottom 3 rows of Ax are zero.
    if (config_.Ax.size() > 0) {
      const pinocchio::SE3 targetPose = getTargetPose(time);
      const ocs2::VectorFunctionLinearApproximation positionApprox = frameDynamicsPtr_->getPositionLinearApproximation(state).front();
      const ocs2::VectorFunctionLinearApproximation orientationApprox =
        frameDynamicsPtr_->getOrientationErrorLinearApproximation(state, {ocs2::matrixToQuaternion(targetPose.rotation())}).front();

      ocs2::VectorFunctionLinearApproximation poseApprox =
        ocs2::VectorFunctionLinearApproximation::Zero(6, state.size(), input.size());
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
      const ocs2::VectorFunctionLinearApproximation velocityApprox = frameDynamicsPtr_->getTwistLinearApproximation(state, input).front();
      linearApproximation.f.segment(row, config_.Av.rows()).noalias() =
        config_.Av * (velocityApprox.f - getTargetTwist(time));
      linearApproximation.dfdx.middleRows(row, config_.Av.rows()).noalias() = config_.Av * velocityApprox.dfdx;
      linearApproximation.dfdu.middleRows(row, config_.Av.rows()).noalias() = config_.Av * velocityApprox.dfdu;
      row += config_.Av.rows();
    }

    if (config_.Aa.size() > 0) {
      const ocs2::VectorFunctionLinearApproximation accelApprox = frameDynamicsPtr_->getAccelerationsLinearApproximation(state, input).front();
      linearApproximation.f.segment(row, config_.Aa.rows()).noalias() =
        config_.Aa * (accelApprox.f - getTargetAcc(time));
      linearApproximation.dfdx.middleRows(row, config_.Aa.rows()).noalias() = config_.Aa * accelApprox.dfdx;
      linearApproximation.dfdu.middleRows(row, config_.Aa.rows()).noalias() = config_.Aa * accelApprox.dfdu;
    }

    return linearApproximation;
  }

}
