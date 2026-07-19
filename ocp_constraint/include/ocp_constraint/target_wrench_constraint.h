#pragma once

#include <memory>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/solver/state_converter.h>
#include <ocp_solver/solver/switched_model_reference_manager.h>
#include <ocp_solver/solver/trajectory.h>

namespace ocp_constraint {

  class TargetWrenchConstraint final : public ocs2::StateInputConstraint {
  public:
    struct Config {
      ocs2::matrix_t A;
      bool useReferenceInputTarget;

      Config() : useReferenceInputTarget(false) {}
    };

    TargetWrenchConstraint(size_t contactIndex,
                           const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                           ocp_solver::WrenchTrajectory targetWrenchTrajectory,
                           Config config = Config());
    TargetWrenchConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                           size_t contactIndex,
                           const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                           ocp_solver::WrenchTrajectory targetWrenchTrajectory,
                           Config config = Config());

    ~TargetWrenchConstraint() override = default;
    TargetWrenchConstraint* clone() const override { return new TargetWrenchConstraint(*this); }

    bool isActive(ocs2::scalar_t time) const override;
    size_t getNumConstraints(ocs2::scalar_t time) const override;
    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation& preComp) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::vector_t& input,
                                                                   const ocs2::PreComputation& preComp) const override;

  private:
    TargetWrenchConstraint(const TargetWrenchConstraint& rhs);

    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_ = nullptr;
    std::unique_ptr<ocp_solver::StateConverter<ocs2::scalar_t>> stateConverterPtr_;
    const size_t contactIndex_;
    ocp_solver::WrenchTrajectory targetWrenchTrajectory_;
    Config config_;
    static constexpr int wrench_dim = 6;
  };

}
