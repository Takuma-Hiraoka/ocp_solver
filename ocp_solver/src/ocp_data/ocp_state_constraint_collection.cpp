#include "ocp_solver/ocp_data/ocp_state_constraint_collection.h"

namespace ocp_solver {

  StateConstraintCollection::StateConstraintCollection(const size_t& ss)
    : state_variable_dim_(ss) {}

  StateConstraintCollection* StateConstraintCollection::clone() const {
    return new StateConstraintCollection(*this);
  }

  ocs2::VectorFunctionLinearApproximation StateConstraintCollection::getLinearApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                                            const ocs2::PreComputation& preComp) const {
    ocs2::VectorFunctionLinearApproximation linearApproximation(getNumConstraints(time), state_variable_dim_);

    // append linearApproximation of each constraintTerm
    size_t i = 0;
    for (const std::unique_ptr<ocs2::StateConstraint>& constraintTerm : this->terms_) {
      if (constraintTerm->isActive(time)) {
        const ocs2::VectorFunctionLinearApproximation constraintTermApproximation = constraintTerm->getLinearApproximation(time, state, preComp);
        const size_t nc = constraintTermApproximation.f.rows();
        linearApproximation.f.segment(i, nc) = constraintTermApproximation.f;
        linearApproximation.dfdx.middleRows(i, nc) = constraintTermApproximation.dfdx;
        i += nc;
      }
    }

    return linearApproximation;
  }

  /******************************************************************************************************/
  /******************************************************************************************************/
  /******************************************************************************************************/
  ocs2::VectorFunctionQuadraticApproximation StateConstraintCollection::getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                                                  const ocs2::PreComputation& preComp) const {
    const size_t numConstraints = getNumConstraints(time);

    ocs2::VectorFunctionQuadraticApproximation quadraticApproximation;
    quadraticApproximation.f.resize(numConstraints);
    quadraticApproximation.dfdx.resize(numConstraints, state_variable_dim_);
    quadraticApproximation.dfdxx.reserve(numConstraints);  // Use reserve instead of resize to avoid unnecessary allocations.

    // append quadraticApproximation of each constraintTerm
    size_t i = 0;
    for (const std::unique_ptr<ocs2::StateConstraint>& constraintTerm : this->terms_) {
      if (constraintTerm->isActive(time)) {
        ocs2::VectorFunctionQuadraticApproximation constraintTermApproximation = constraintTerm->getQuadraticApproximation(time, state, preComp);
        const size_t nc = constraintTermApproximation.f.rows();
        quadraticApproximation.f.segment(i, nc) = constraintTermApproximation.f;
        quadraticApproximation.dfdx.middleRows(i, nc) = constraintTermApproximation.dfdx;
        ocs2::appendVectorToVectorByMoving(quadraticApproximation.dfdxx, std::move(constraintTermApproximation.dfdxx));
        i += nc;
      }
    }

    return quadraticApproximation;
  }

}
