#pragma once
#include <ocs2_core/cost/StateInputCostCollection.h>

namespace ocp_solver {
  class StateInputCostCollection : public ocs2::StateInputCostCollection {
  public:
    StateInputCostCollection(const size_t& ss, const size_t& is);
    ~StateInputCostCollection() override = default;
    StateInputCostCollection* clone() const override;

    ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input,
                                                                                 const ocs2::TargetTrajectories& targetTrajectories,
                                                                                 const ocs2::PreComputation& preComp) const override;

  private:
    const size_t state_variable_dim_;
    const size_t input_variable_dim_;
  };

}
