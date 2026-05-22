#include "ocp_solver/ocp_data/ocp_state_input_constraint_collection.h"

namespace ocp_solver {

  StateInputConstraintCollection::StateInputConstraintCollection(const size_t& ss, const size_t& is)
    : state_variable_dim_(ss),
      input_variable_dim_(is) {}

  StateInputConstraintCollection* StateInputConstraintCollection::clone() const {
    return new StateInputConstraintCollection(*this);
  }

  ocs2::VectorFunctionLinearApproximation StateInputConstraintCollection::getLinearApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                                                 const ocs2::vector_t& input,
                                                                                                 const ocs2::PreComputation& preComp) const {

    ocs2::VectorFunctionLinearApproximation linearApproximation(getNumConstraints(time), state_variable_dim_, input_variable_dim_);

    // append linearApproximation of each constraintTerm
    size_t i = 0;
    for (const std::unique_ptr<ocs2::StateInputConstraint>& constraintTerm : this->terms_) {
      if (constraintTerm->isActive(time)) {
        const ocs2::VectorFunctionLinearApproximation constraintTermApproximation = constraintTerm->getLinearApproximation(time, state, input, preComp);
        const size_t nc = constraintTermApproximation.f.rows();
        linearApproximation.f.segment(i, nc) = constraintTermApproximation.f;
        linearApproximation.dfdx.middleRows(i, nc) = constraintTermApproximation.dfdx;
        linearApproximation.dfdu.middleRows(i, nc) = constraintTermApproximation.dfdu;
        i += nc;
      }
    }

    return linearApproximation;
  }

  ocs2::VectorFunctionQuadraticApproximation StateInputConstraintCollection::getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                                                       const ocs2::vector_t& input,
                                                                                                       const ocs2::PreComputation& preComp) const {
    const size_t numConstraints = getNumConstraints(time);

    ocs2::VectorFunctionQuadraticApproximation quadraticApproximation;
    quadraticApproximation.f.resize(numConstraints);
    quadraticApproximation.dfdx.resize(numConstraints, state_variable_dim_);
    quadraticApproximation.dfdu.resize(numConstraints, input_variable_dim_);
    quadraticApproximation.dfdxx.reserve(numConstraints);  // Use reserve instead of resize to avoid unnecessary allocations.
    quadraticApproximation.dfdux.reserve(numConstraints);
    quadraticApproximation.dfduu.reserve(numConstraints);

    // append quadraticApproximation of each constraintTerm
    size_t i = 0;
    for (const std::unique_ptr<ocs2::StateInputConstraint>& constraintTerm : this->terms_) {
      if (constraintTerm->isActive(time)) {
        ocs2::VectorFunctionQuadraticApproximation constraintTermApproximation = constraintTerm->getQuadraticApproximation(time, state, input, preComp);
        const size_t nc = constraintTermApproximation.f.rows();
        quadraticApproximation.f.segment(i, nc) = constraintTermApproximation.f;
        quadraticApproximation.dfdx.middleRows(i, nc) = constraintTermApproximation.dfdx;
        quadraticApproximation.dfdu.middleRows(i, nc) = constraintTermApproximation.dfdu;
        ocs2::appendVectorToVectorByMoving(quadraticApproximation.dfdxx, std::move(constraintTermApproximation.dfdxx));
        ocs2::appendVectorToVectorByMoving(quadraticApproximation.dfdux, std::move(constraintTermApproximation.dfdux));
        ocs2::appendVectorToVectorByMoving(quadraticApproximation.dfduu, std::move(constraintTermApproximation.dfduu));
        i += nc;
      }
    }

    return quadraticApproximation;
  }

}
