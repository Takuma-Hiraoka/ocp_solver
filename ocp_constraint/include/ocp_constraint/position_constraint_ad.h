#pragma once

#include <memory>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/pinocchio_endeffector_dynamics_cppad.h>

namespace ocp_constraint {

  class PositionConstraintAD final : public ocs2::StateInputConstraint {
  public:
    struct Config {
      ocs2::matrix_t Ax;
      ocs2::matrix_t Av;
      ocs2::matrix_t Aa;
    };

    PositionConstraintAD(const ocp_solver::PinocchioEndEffectorDynamicsCppAd& endEffectorDynamics,
                         const pinocchio::SE3 targetPose,
                         size_t numConstraints,
                         Config config = Config());

    ~PositionConstraintAD() override = default;
    PositionConstraintAD* clone() const override { return new PositionConstraintAD(*this); }

    void configure(Config&& config);
    void configure(const Config& config) { this->configure(Config(config)); }

    ocp_solver::PinocchioEndEffectorDynamicsCppAd& getEndEffectorDynamics() { return *endEffectorDynamicsPtr_; }

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
    PositionConstraintAD(const PositionConstraintAD& rhs);
    std::unique_ptr<ocp_solver::PinocchioEndEffectorDynamicsCppAd> endEffectorDynamicsPtr_;
    pinocchio::SE3 targetPose_;
    const size_t numConstraints_;
    Config config_;
  };

}
