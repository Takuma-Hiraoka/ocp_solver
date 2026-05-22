#include "ocp_constraint/position_constraint.h"
#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocp_constraint {
  PositionConstraint::PositionConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                         const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                                         size_t numConstraints,
                                         Config config)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      frameDynamicsPtr_(frameDynamics.clone()),
      numConstraints_(numConstraints),
      config_(std::move(config)) {}

  PositionConstraint::PositionConstraint(const PositionConstraint& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      frameDynamicsPtr_(rhs.frameDynamicsPtr_->clone()),
      numConstraints_(rhs.numConstraints_),
      config_(rhs.config_) {}

  void PositionConstraint::configure(Config&& config) {
    assert(config.Ax.size() > 0 || config.Av.size() > 0);
    assert((config.Ax.size() > 0 && config.Ax.rows() == numConstraints_) || config.Ax.size() == 0);
    assert((config.Ax.size() > 0 && config.Ax.cols() == 6) || config.Ax.size() == 0);
    assert((config.Av.size() > 0 && config.Av.rows() == numConstraints_) || config.Av.size() == 0);
    assert((config.Av.size() > 0 && config.Av.cols() == 6) || config.Av.size() == 0);
    assert((config.Aa.size() > 0 && config.Aa.rows() == numConstraints_) || config.Aa.size() == 0);
    assert((config.Aa.size() > 0 && config.Aa.cols() == 6) || config.Aa.size() == 0);
    config_ = std::move(config);
  }

  bool PositionConstraint::isActive(ocs2::scalar_t time) const {
    return referenceManagerPtr_->isInContact(time, frameDynamicsPtr_->getFrameId());
  }

  ocs2::vector_t PositionConstraint::getValue(ocs2::scalar_t time,
                                              const ocs2::vector_t& state,
                                              const ocs2::vector_t& input,
                                              const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::vector_t f = ocs2::vector_t::Zero(numConstraints_);
    if (config_.Ax.size() > 0) {
      pinocchio::SE3 targetPose = pinocchio::SE3::Identity();
      for (const std::pair<pinocchio::FrameIndex, pinocchio::SE3> contact : referenceManagerPtr_->getContacts(time)) {
        if (contact.first == frameDynamicsPtr_->getFrameId()) targetPose = contact.second;
      }
      Eigen::Matrix<ocs2::scalar_t, 6, 1> xError;
      xError << frameDynamicsPtr_->getPosition(ocpPreComp) - targetPose.translation(),
        frameDynamicsPtr_->getOrientationError(ocpPreComp, ocs2::matrixToQuaternion(targetPose.rotation()));
      f.noalias() += config_.Ax * xError;
    }
    if (config_.Av.size() > 0) {
      f.noalias() += config_.Av * frameDynamicsPtr_->getTwist(ocpPreComp);
    }
    if (config_.Aa.size() > 0) {
      f.noalias() += config_.Aa * frameDynamicsPtr_->getAccelerations(ocpPreComp);
    }
    return f;
  }

  ocs2::VectorFunctionLinearApproximation PositionConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                     const ocs2::vector_t& state,
                                                                                     const ocs2::vector_t& input,
                                                                                     const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    ocs2::VectorFunctionLinearApproximation linearApproximation =
      ocs2::VectorFunctionLinearApproximation::Zero(getNumConstraints(time), 2*pinocchioInterface.getModel().nv, input.size());


    if (config_.Ax.size() > 0) {
      pinocchio::SE3 targetPose = pinocchio::SE3::Identity();
      for (const std::pair<pinocchio::FrameIndex, pinocchio::SE3> contact : referenceManagerPtr_->getContacts(time)) {
        if (contact.first == frameDynamicsPtr_->getFrameId()) targetPose = contact.second;
      }
      const ocs2::VectorFunctionLinearApproximation positionApprox = frameDynamicsPtr_->getPositionLinearApproximation(ocpPreComp);
      const ocs2::VectorFunctionLinearApproximation orientationApprox =
        frameDynamicsPtr_->getOrientationErrorLinearApproximation(ocpPreComp, ocs2::matrixToQuaternion(targetPose.rotation()));

      linearApproximation.f.head(3).noalias() += config_.Ax.topLeftCorner(3, 3) * (positionApprox.f - targetPose.translation());
      linearApproximation.f.tail(3).noalias() += config_.Ax.bottomRightCorner(3, 3) * orientationApprox.f;
      linearApproximation.dfdx.topRows(3).noalias() += config_.Ax.topLeftCorner(3, 3) * positionApprox.dfdx;
      linearApproximation.dfdx.bottomRows(3).noalias() += config_.Ax.bottomRightCorner(3, 3) * orientationApprox.dfdx;
    }

    if (config_.Av.size() > 0) {
      const ocs2::VectorFunctionLinearApproximation velocityApprox = frameDynamicsPtr_->getTwistLinearApproximation(ocpPreComp);
      linearApproximation.f.noalias() += config_.Av * velocityApprox.f;
      linearApproximation.dfdx.noalias() += config_.Av * velocityApprox.dfdx;
      linearApproximation.dfdu.noalias() += config_.Av * velocityApprox.dfdu;
    }

    if (config_.Aa.size() > 0) {
      const ocs2::VectorFunctionLinearApproximation accelApprox = frameDynamicsPtr_->getAccelerationsLinearApproximation(ocpPreComp);
      linearApproximation.f.noalias() += config_.Aa * accelApprox.f;
      linearApproximation.dfdx.noalias() += config_.Aa * accelApprox.dfdx;
      linearApproximation.dfdu.noalias() += config_.Aa * accelApprox.dfdu;
    }

    return linearApproximation;
  }

}
