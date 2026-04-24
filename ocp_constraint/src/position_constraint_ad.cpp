#include "ocp_constraint/position_constraint_ad.h"
#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocp_constraint {
  PositionConstraintAD::PositionConstraintAD(const ocp_solver::PinocchioEndEffectorDynamicsCppAd& endEffectorDynamics,
                                             const pinocchio::SE3 targetPose,
                                             size_t numConstraints,
                                             Config config)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      endEffectorDynamicsPtr_(endEffectorDynamics.clone()),
      targetPose_(targetPose),
      numConstraints_(numConstraints),
      config_(std::move(config)) {}

  PositionConstraintAD::PositionConstraintAD(const PositionConstraintAD& rhs)
    : StateInputConstraint(rhs),
      endEffectorDynamicsPtr_(rhs.endEffectorDynamicsPtr_->clone()),
      targetPose_(rhs.targetPose_),
      numConstraints_(rhs.numConstraints_),
      config_(rhs.config_) {}

  void PositionConstraintAD::configure(Config&& config) {
    assert(config.Ax.size() > 0 || config.Av.size() > 0);
    assert((config.Ax.size() > 0 && config.Ax.rows() == numConstraints_) || config.Ax.size() == 0);
    assert((config.Ax.size() > 0 && config.Ax.cols() == 6) || config.Ax.size() == 0);
    assert((config.Av.size() > 0 && config.Av.rows() == numConstraints_) || config.Av.size() == 0);
    assert((config.Av.size() > 0 && config.Av.cols() == 6) || config.Av.size() == 0);
    assert((config.Aa.size() > 0 && config.Aa.rows() == numConstraints_) || config.Aa.size() == 0);
    assert((config.Aa.size() > 0 && config.Aa.cols() == 6) || config.Aa.size() == 0);
    config_ = std::move(config);
  }

  ocs2::vector_t PositionConstraintAD::getValue(ocs2::scalar_t time,
                                                const ocs2::vector_t& state,
                                                const ocs2::vector_t& input,
                                                const ocs2::PreComputation& preComp) const {
    ocs2::vector_t f = ocs2::vector_t::Zero(numConstraints_);
    if (config_.Ax.size() > 0) {
      // foot pose is a 6D vector containing the foot position and orientation error wrt. to the ground normal
      Eigen::Matrix<ocs2::scalar_t, 6, 1> xError;
      xError << endEffectorDynamicsPtr_->getPosition(state).front() - targetPose_.translation(),
        endEffectorDynamicsPtr_->getOrientationError(state, {ocs2::matrixToQuaternion(targetPose_.rotation())}).front();
      f.noalias() += config_.Ax * xError;
    }
    if (config_.Av.size() > 0) {
      f.noalias() += config_.Av * endEffectorDynamicsPtr_->getTwist(state, input).front();
    }
    if (config_.Aa.size() > 0) {
      f.noalias() += config_.Aa * endEffectorDynamicsPtr_->getAccelerations(state, input).front();
    }
    return f;
  }

  ocs2::VectorFunctionLinearApproximation PositionConstraintAD::getLinearApproximation(ocs2::scalar_t time,
                                                                                       const ocs2::vector_t& state,
                                                                                       const ocs2::vector_t& input,
                                                                                       const ocs2::PreComputation& preComp) const {
    ocs2::VectorFunctionLinearApproximation linearApproximation =
      ocs2::VectorFunctionLinearApproximation::Zero(getNumConstraints(time), state.size(), input.size());

    // Orientation error gains are ignored for now
    // This is equal with assuming that the bottom 3 rows of Ax are zero.
    if (config_.Ax.size() > 0) {
      const auto positionApprox = endEffectorDynamicsPtr_->getPositionLinearApproximation(state).front();
      const auto orientationApprox =
        endEffectorDynamicsPtr_->getOrientationErrorLinearApproximation(state, {ocs2::matrixToQuaternion(targetPose_.rotation())}).front();

      linearApproximation.f.head(3).noalias() += config_.Ax.topLeftCorner(3, 3) * (positionApprox.f - targetPose_.translation());
      linearApproximation.f.tail(3).noalias() += config_.Ax.bottomRightCorner(3, 3) * orientationApprox.f;
      linearApproximation.dfdx.topRows(3).noalias() += config_.Ax.topLeftCorner(3, 3) * positionApprox.dfdx;
      linearApproximation.dfdx.bottomRows(3).noalias() += config_.Ax.bottomRightCorner(3, 3) * orientationApprox.dfdx;
    }

    if (config_.Av.size() > 0) {
      const auto velocityApprox = endEffectorDynamicsPtr_->getTwistLinearApproximation(state, input).front();
      linearApproximation.f.noalias() += config_.Av * velocityApprox.f;
      linearApproximation.dfdx.noalias() += config_.Av * velocityApprox.dfdx;
      linearApproximation.dfdu.noalias() += config_.Av * velocityApprox.dfdu;
    }

    if (config_.Aa.size() > 0) {
      const auto accelApprox = endEffectorDynamicsPtr_->getAccelerationsLinearApproximation(state, input).front();
      linearApproximation.f.noalias() += config_.Aa * accelApprox.f;
      linearApproximation.dfdx.noalias() += config_.Aa * accelApprox.dfdx;
      linearApproximation.dfdu.noalias() += config_.Aa * accelApprox.dfdu;
    }

    return linearApproximation;
  }

}
