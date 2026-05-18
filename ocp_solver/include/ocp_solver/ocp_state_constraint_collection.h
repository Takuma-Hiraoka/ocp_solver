#pragma once

#include <ocs2_core/constraint/StateConstraintCollection.h>

namespace ocp_solver {

  class StateConstraintCollection : public ocs2::StateConstraintCollection {
  public:
    StateConstraintCollection(const size_t& ss);
    ~StateConstraintCollection() override = default;
    StateConstraintCollection* clone() const override;

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                     const ocs2::PreComputation& preComp) const;

    ocs2::VectorFunctionQuadraticApproximation getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                           const ocs2::PreComputation& preComp) const;

  private:
    const size_t state_variable_dim_;
  };

}
