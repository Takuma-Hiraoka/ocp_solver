#include "ocp_solver/ocp_data/ocp_state_constraint_collection.h"

#include <stdexcept>
#include <vector>

namespace ocp_solver {
  namespace {
    void checkApproximationDimensions(const ocs2::VectorFunctionLinearApproximation& approximation,
                                      size_t stateVariableDim) {
      if (approximation.dfdx.rows() != approximation.f.rows()
          || approximation.dfdx.cols() != static_cast<Eigen::Index>(stateVariableDim)) {
        throw std::runtime_error("StateConstraintCollection received a linear approximation with inconsistent dimensions.");
      }
    }

    void checkApproximationDimensions(const ocs2::VectorFunctionQuadraticApproximation& approximation,
                                      size_t stateVariableDim) {
      if (approximation.dfdx.rows() != approximation.f.rows()
          || approximation.dfdx.cols() != static_cast<Eigen::Index>(stateVariableDim)
          || approximation.dfdxx.size() != static_cast<size_t>(approximation.f.rows())) {
        throw std::runtime_error("StateConstraintCollection received a quadratic approximation with inconsistent dimensions.");
      }
    }
  }  // namespace

  StateConstraintCollection::StateConstraintCollection(const size_t& ss)
    : state_variable_dim_(ss) {}

  StateConstraintCollection* StateConstraintCollection::clone() const {
    return new StateConstraintCollection(*this);
  }

  ocs2::VectorFunctionLinearApproximation StateConstraintCollection::getLinearApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                                            const ocs2::PreComputation& preComp) const {
    std::vector<ocs2::VectorFunctionLinearApproximation> activeApproximations;
    size_t numConstraints = 0;
    for (const std::unique_ptr<ocs2::StateConstraint>& constraintTerm : this->terms_) {
      if (constraintTerm->isActive(time)) {
        activeApproximations.push_back(constraintTerm->getLinearApproximation(time, state, preComp));
        checkApproximationDimensions(activeApproximations.back(), state_variable_dim_);
        numConstraints += static_cast<size_t>(activeApproximations.back().f.rows());
      }
    }

    ocs2::VectorFunctionLinearApproximation linearApproximation =
      ocs2::VectorFunctionLinearApproximation::Zero(numConstraints, state_variable_dim_);

    size_t i = 0;
    for (const ocs2::VectorFunctionLinearApproximation& constraintTermApproximation : activeApproximations) {
      const size_t nc = static_cast<size_t>(constraintTermApproximation.f.rows());
      linearApproximation.f.segment(i, nc) = constraintTermApproximation.f;
      linearApproximation.dfdx.middleRows(i, nc) = constraintTermApproximation.dfdx;
      i += nc;
    }
    return linearApproximation;
  }

  /******************************************************************************************************/
  /******************************************************************************************************/
  /******************************************************************************************************/
  ocs2::VectorFunctionQuadraticApproximation StateConstraintCollection::getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                                                  const ocs2::PreComputation& preComp) const {
    std::vector<ocs2::VectorFunctionQuadraticApproximation> activeApproximations;
    size_t numConstraints = 0;
    for (const std::unique_ptr<ocs2::StateConstraint>& constraintTerm : this->terms_) {
      if (constraintTerm->isActive(time)) {
        activeApproximations.push_back(constraintTerm->getQuadraticApproximation(time, state, preComp));
        checkApproximationDimensions(activeApproximations.back(), state_variable_dim_);
        numConstraints += static_cast<size_t>(activeApproximations.back().f.rows());
      }
    }

    ocs2::VectorFunctionQuadraticApproximation quadraticApproximation;
    quadraticApproximation.f.setZero(numConstraints);
    quadraticApproximation.dfdx.setZero(numConstraints, state_variable_dim_);
    quadraticApproximation.dfdxx.reserve(numConstraints);  // Use reserve instead of resize to avoid unnecessary allocations.

    size_t i = 0;
    for (ocs2::VectorFunctionQuadraticApproximation& constraintTermApproximation : activeApproximations) {
      const size_t nc = static_cast<size_t>(constraintTermApproximation.f.rows());
      quadraticApproximation.f.segment(i, nc) = constraintTermApproximation.f;
      quadraticApproximation.dfdx.middleRows(i, nc) = constraintTermApproximation.dfdx;
      ocs2::appendVectorToVectorByMoving(quadraticApproximation.dfdxx, std::move(constraintTermApproximation.dfdxx));
      i += nc;
    }

    return quadraticApproximation;
  }

}
