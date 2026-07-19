#include "ocp_constraint/target_wrench_constraint.h"

#include <ocp_solver/common/scope_profiler.h>

#include <utility>

namespace ocp_constraint {

  TargetWrenchConstraint::TargetWrenchConstraint(
      size_t contactIndex,
      const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
      ocp_solver::WrenchTrajectory targetWrenchTrajectory,
      Config config)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      stateConverterPtr_(stateConverter.clone()),
      contactIndex_(contactIndex),
      targetWrenchTrajectory_(std::move(targetWrenchTrajectory)),
      config_(std::move(config)) {
    if (config_.A.size() == 0) {
      config_.A.setIdentity(wrench_dim, wrench_dim);
    }
  }

  TargetWrenchConstraint::TargetWrenchConstraint(
      const ocp_solver::SwitchedModelReferenceManager& referenceManager,
      size_t contactIndex,
      const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
      ocp_solver::WrenchTrajectory targetWrenchTrajectory,
      Config config)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      stateConverterPtr_(stateConverter.clone()),
      contactIndex_(contactIndex),
      targetWrenchTrajectory_(std::move(targetWrenchTrajectory)),
      config_(std::move(config)) {
    if (config_.A.size() == 0) {
      config_.A.setIdentity(wrench_dim, wrench_dim);
    }
  }

  TargetWrenchConstraint::TargetWrenchConstraint(const TargetWrenchConstraint& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      stateConverterPtr_(rhs.stateConverterPtr_->clone()),
      contactIndex_(rhs.contactIndex_),
      targetWrenchTrajectory_(rhs.targetWrenchTrajectory_),
      config_(rhs.config_) {}

  bool TargetWrenchConstraint::isActive(ocs2::scalar_t time) const {
    if (referenceManagerPtr_ == nullptr) return true;
    return referenceManagerPtr_->isInContact(time, stateConverterPtr_->getContactCandidateIds()[contactIndex_]);
  }

  size_t TargetWrenchConstraint::getNumConstraints(ocs2::scalar_t time) const {
    return static_cast<size_t>(config_.A.rows());
  }

  ocs2::vector_t TargetWrenchConstraint::getValue(ocs2::scalar_t time,
                                                  const ocs2::vector_t& state,
                                                  const ocs2::vector_t& input,
                                                  const ocs2::PreComputation& preComp) const {
    OCP_SOLVER_PROFILE_SCOPE("TargetWrenchConstraint::getValue");
    ocs2::vector_t targetWrench = targetWrenchTrajectory_.getTargetWrench(time);
    if (config_.useReferenceInputTarget && referenceManagerPtr_ != nullptr) {
      targetWrench = stateConverterPtr_->getContactWrench(
          referenceManagerPtr_->getTargetTrajectories().getDesiredInput(time), contactIndex_);
    }
    return config_.A * (stateConverterPtr_->getContactWrench(input, contactIndex_) - targetWrench);
  }

  ocs2::VectorFunctionLinearApproximation TargetWrenchConstraint::getLinearApproximation(
      ocs2::scalar_t time,
      const ocs2::vector_t& state,
      const ocs2::vector_t& input,
      const ocs2::PreComputation& preComp) const {
    OCP_SOLVER_PROFILE_SCOPE("TargetWrenchConstraint::getLinearApproximation");
    ocs2::VectorFunctionLinearApproximation approx;
    approx.f = getValue(time, state, input, preComp);
    approx.dfdx = ocs2::matrix_t::Zero(getNumConstraints(time), stateConverterPtr_->getStateVariableDim());
    approx.dfdu = ocs2::matrix_t::Zero(getNumConstraints(time), stateConverterPtr_->getInputDim());
    approx.dfdu.block(0, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_),
                      config_.A.rows(), wrench_dim) = config_.A;
    return approx;
  }

}
