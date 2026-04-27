#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>
#include "ocp_solver/state_converter.h"
#include "ocp_solver/switched_model_reference_manager.h"

namespace ocp_solver {

  class ZeroWrenchConstraint final : public ocs2::StateInputConstraint {
  public:
    ZeroWrenchConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                         size_t contactIndex,
                         const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter);

    ~ZeroWrenchConstraint() override = default;
    ZeroWrenchConstraint* clone() const override { return new ZeroWrenchConstraint(*this); }

    bool isActive(ocs2::scalar_t time) const override;
    size_t getNumConstraints(ocs2::scalar_t time) const override { return n_constraints; }
    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation& preComp) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::vector_t& input,
                                                                   const ocs2::PreComputation& preComp) const override;

  private:
    ZeroWrenchConstraint(const ZeroWrenchConstraint& rhs);
    const ocp_solver::StateConverter<ocs2::scalar_t>* stateConverterPtr_;
    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_;
    const size_t contactIndex_;
    static const int n_constraints = 6;
  };

}
