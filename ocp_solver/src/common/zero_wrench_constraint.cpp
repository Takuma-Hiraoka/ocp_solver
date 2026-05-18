#include "ocp_solver/common/zero_wrench_constraint.h"

namespace ocp_solver {

  ZeroWrenchConstraint::ZeroWrenchConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                             size_t contactIndex,
                                             const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      contactIndex_(contactIndex),
      stateConverterPtr_(&stateConverter) {}

  ZeroWrenchConstraint::ZeroWrenchConstraint(const ZeroWrenchConstraint& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      contactIndex_(rhs.contactIndex_),
      stateConverterPtr_(rhs.stateConverterPtr_->clone()) {}

  bool ZeroWrenchConstraint::isActive(ocs2::scalar_t time) const {
    return !referenceManagerPtr_->isInContact(time, stateConverterPtr_->getContactCandidateIds()[contactIndex_]);
  }

  ocs2::vector_t ZeroWrenchConstraint::getValue(ocs2::scalar_t time,
                                                const ocs2::vector_t& state,
                                                const ocs2::vector_t& input,
                                                const ocs2::PreComputation& preComp) const {
    return stateConverterPtr_->getContactWrench(input, contactIndex_);
  }

  ocs2::VectorFunctionLinearApproximation ZeroWrenchConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                       const ocs2::vector_t& state,
                                                                                       const ocs2::vector_t& input,
                                                                                       const ocs2::PreComputation& preComp) const {
    ocs2::VectorFunctionLinearApproximation approx;
    approx.f = getValue(time, state, input, preComp);
    approx.dfdx = ocs2::matrix_t::Zero(n_constraints, stateConverterPtr_->getStateVariableDim());
    approx.dfdu = ocs2::matrix_t::Zero(n_constraints, stateConverterPtr_->getInputDim());
    approx.dfdu.middleCols<n_constraints>(n_constraints * contactIndex_).diagonal() = ocs2::vector_t::Ones(n_constraints);
    return approx;
  }

}
