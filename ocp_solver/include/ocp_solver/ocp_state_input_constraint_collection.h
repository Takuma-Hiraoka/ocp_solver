#pragma once

#include <ocs2_core/constraint/StateInputConstraintCollection.h>

namespace ocp_solver {

  class StateInputConstraintCollection : public ocs2::StateInputConstraintCollection {
  public:
    StateInputConstraintCollection(const size_t& ss, const size_t& is);
    ~StateInputConstraintCollection() override = default;
    StateInputConstraintCollection* clone() const override;

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input,
                                                                     const ocs2::PreComputation& preComp) const;

    ocs2::VectorFunctionQuadraticApproximation getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input,
                                                                           const ocs2::PreComputation& preComp) const;

  private:
    const size_t state_variable_dim_;
    const size_t input_variable_dim_;
  };

}
