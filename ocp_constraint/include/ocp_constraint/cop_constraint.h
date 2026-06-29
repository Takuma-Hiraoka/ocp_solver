#pragma once

#include <memory>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/pinocchio/pinocchio_frame_dynamics.h>
#include <ocp_solver/solver/state_converter.h>
#include <ocp_solver/solver/switched_model_reference_manager.h>

namespace ocp_constraint {

class CopConstraint final : public ocs2::StateInputConstraint {
 public:
  struct Config {
    Config() {}
    ocs2::scalar_t normalForceRegularization = 1.0;
  };

  CopConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                size_t contactIndex,
                const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                Config config = Config());

  ~CopConstraint() override = default;
  CopConstraint* clone() const override { return new CopConstraint(*this); }

  bool isActive(ocs2::scalar_t time) const override;
  size_t getNumConstraints(ocs2::scalar_t time) const override { return 3; }
  ocs2::vector_t getValue(ocs2::scalar_t time,
                          const ocs2::vector_t& state,
                          const ocs2::vector_t& input,
                          const ocs2::PreComputation& preComp) const override;
  ocs2::VectorFunctionLinearApproximation getLinearApproximation(
      ocs2::scalar_t time,
      const ocs2::vector_t& state,
      const ocs2::vector_t& input,
      const ocs2::PreComputation& preComp) const override;

 private:
  CopConstraint(const CopConstraint& rhs);

  size_t contactIndex_ = 0;
  Config config_;
  std::unique_ptr<ocp_solver::StateConverter<ocs2::scalar_t>> stateConverterPtr_;
  std::unique_ptr<ocp_solver::PinocchioFrameDynamics> frameDynamicsPtr_;
  const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_ = nullptr;
};

}  // namespace ocp_constraint
