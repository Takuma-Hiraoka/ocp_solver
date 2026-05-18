#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/pinocchio/pinocchio_frame_dynamics.h>
#include <ocp_solver/solver/switched_model_reference_manager.h>

namespace ocp_constraint {

  class PositionConstraint final : public ocs2::StateInputConstraint {
  public:
    struct Config {
      ocs2::matrix_t Ax;
      ocs2::matrix_t Av;
      ocs2::matrix_t Aa;
    };

    PositionConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                       const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                       size_t numConstraints,
                       Config config = Config());

    ~PositionConstraint() override = default;
    PositionConstraint* clone() const override { return new PositionConstraint(*this); }

    void configure(Config&& config);
    void configure(const Config& config) { this->configure(Config(config)); }

    ocp_solver::PinocchioFrameDynamics& getFrameDynamics() { return *frameDynamicsPtr_; }

    bool isActive(ocs2::scalar_t time) const override;
    size_t getNumConstraints(ocs2::scalar_t time) const override { return numConstraints_; }
    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation& preComp) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::vector_t& input,
                                                                   const ocs2::PreComputation& preComp) const override;

  private:
    PositionConstraint(const PositionConstraint& rhs);
    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_;
    std::unique_ptr<ocp_solver::PinocchioFrameDynamics> frameDynamicsPtr_;
    const size_t numConstraints_;
    Config config_;
  };
}
