#pragma once

#include <ocs2_core/cost/StateCostCollection.h>

namespace ocp_solver {

  class StateCostCollection : public ocs2::StateCostCollection {
  public:
    StateCostCollection(const size_t& ss);
    ~StateCostCollection() override = default;
    StateCostCollection* clone() const override;

    /** Get state-only cost quadratic approximation */
    ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                         const ocs2::TargetTrajectories& targetTrajectories,
                                                                         const ocs2::PreComputation& preComp) const override;

  protected:
    const size_t state_variable_dim_;
  };

}
