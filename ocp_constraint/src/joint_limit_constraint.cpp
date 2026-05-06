#include "ocp_constraint/joint_limit_constraint.h"

#include <iostream>

namespace ocp_constraint {
  JointLimitsConstraint::JointLimitsConstraint(const ocs2::PinocchioInterface pinocchioInterface,
                                               const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                                               ocs2::PieceWisePolynomialBarrierPenalty::Config barrierConfig)
    : penaltyPtr_(new ocs2::PieceWisePolynomialBarrierPenalty(barrierConfig)),
      stateConverterPtr_(&stateConverter) {
    const pinocchio::Model& model = pinocchioInterface.getModel();
    ocs2::vector_t upper_limits = model.upperPositionLimit.tail(stateConverterPtr_->getJointDim());
    ocs2::vector_t lower_limits = model.lowerPositionLimit.tail(stateConverterPtr_->getJointDim());
    positionLimits_ = {lower_limits, upper_limits};
  }

  JointLimitsConstraint::JointLimitsConstraint(const JointLimitsConstraint& rhs)
    : penaltyPtr_(rhs.penaltyPtr_->clone()),
      positionLimits_(rhs.positionLimits_),
      stateConverterPtr_(rhs.stateConverterPtr_) {}

  ocs2::scalar_t JointLimitsConstraint::getValue(ocs2::scalar_t time,
                                                 const ocs2::vector_t& state,
                                                 const ocs2::TargetTrajectories& targetTrajectories,
                                                 const ocs2::PreComputation& preComp) const {
    return getValue(stateConverterPtr_->getJointAngles(state));
  }

  ocs2::ScalarFunctionQuadraticApproximation JointLimitsConstraint::getQuadraticApproximation(ocs2::scalar_t time,
                                                                                              const ocs2::vector_t& state,
                                                                                              const ocs2::TargetTrajectories& targetTrajectories,
                                                                                              const ocs2::PreComputation& preComp) const {
    return getQuadraticApproximation(stateConverterPtr_->getJointAngles(state));
  }

  ocs2::scalar_t JointLimitsConstraint::getValue(const ocs2::vector_t& jointPositions) const {
    const ocs2::vector_t upperBoundPositionOffset = positionLimits_.second - jointPositions;
    const ocs2::vector_t lowerBoundPositionOffset = jointPositions - positionLimits_.first;

    return upperBoundPositionOffset.unaryExpr([&](ocs2::scalar_t hi) { return penaltyPtr_->getValue(0.0, hi); }).sum() +
      lowerBoundPositionOffset.unaryExpr([&](ocs2::scalar_t hi) { return penaltyPtr_->getValue(0.0, hi); }).sum();
  }

  ocs2::ScalarFunctionQuadraticApproximation JointLimitsConstraint::getQuadraticApproximation(const ocs2::vector_t& jointPositions) const {
    const ocs2::vector_t upperBoundPositionOffset = positionLimits_.second - jointPositions;
    const ocs2::vector_t lowerBoundPositionOffset = jointPositions - positionLimits_.first;

    const size_t stateDim = stateConverterPtr_->getStateVariableDim();
    const size_t jointDim = stateConverterPtr_->getJointDim();
    const size_t jointStartIndex = stateConverterPtr_->getJointStartindex();

    ocs2::ScalarFunctionQuadraticApproximation cost;
    cost.f = upperBoundPositionOffset.unaryExpr([&](ocs2::scalar_t hi) { return penaltyPtr_->getValue(0.0, hi); }).sum() +
      lowerBoundPositionOffset.unaryExpr([&](ocs2::scalar_t hi) { return penaltyPtr_->getValue(0.0, hi); }).sum();

    cost.dfdx = ocs2::vector_t::Zero(stateDim);
    cost.dfdx.segment(jointStartIndex, jointDim) = lowerBoundPositionOffset.unaryExpr([&](ocs2::scalar_t hi) {
                                                                                        return penaltyPtr_->getDerivative(0.0, hi);
                                                                                      }) - upperBoundPositionOffset.unaryExpr([&](ocs2::scalar_t hi) { return penaltyPtr_->getDerivative(0.0, hi); });

    cost.dfdxx = ocs2::matrix_t::Zero(stateDim, stateDim);
    cost.dfdxx.block(jointStartIndex, jointStartIndex, jointDim, jointDim).diagonal() = lowerBoundPositionOffset.unaryExpr([&](ocs2::scalar_t hi) {
                                                                                                                             return penaltyPtr_->getSecondDerivative(0.0, hi);
                                                                                                                           }) + upperBoundPositionOffset.unaryExpr([&](ocs2::scalar_t hi) { return penaltyPtr_->getSecondDerivative(0.0, hi); });

    return cost;
  }

  void JointLimitsConstraint::setGains(const ocs2::scalar_t& mu, const ocs2::scalar_t& delta) {
    penaltyPtr_->setConfig(ocs2::PieceWisePolynomialBarrierPenalty::Config(mu, delta));
  }

  void JointLimitsConstraint::getGains(ocs2::scalar_t& mu, ocs2::scalar_t& delta) const {
    ocs2::PieceWisePolynomialBarrierPenalty::Config config;
    penaltyPtr_->getConfig(config);
    mu = config.mu;
    delta = config.delta;
  }

}
