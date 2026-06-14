#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/pinocchio/pinocchio_frame_dynamics.h>
#include <ocp_solver/solver/switched_model_reference_manager.h>

namespace ocp_constraint {

  class PositionConstraint : public ocs2::StateInputConstraint {
  public:
    struct Config {
      ocs2::matrix_t Ax;
      ocs2::matrix_t Av;
      ocs2::matrix_t Aa;
    };

    PositionConstraint(const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                       size_t numConstraints,
                       Config config = Config(),
                       pinocchio::SE3 targetPose = pinocchio::SE3::Identity(),
                       ocs2::vector_t targetTwist = ocs2::vector_t::Zero(6),
                       ocs2::vector_t targetAcc = ocs2::vector_t::Zero(6));

    ~PositionConstraint() override = default;
    PositionConstraint* clone() const override { return new PositionConstraint(*this); }
    PositionConstraint(const PositionConstraint& rhs);

    void configure(Config&& config);
    void configure(const Config& config) { this->configure(Config(config)); }

    const ocp_solver::PinocchioFrameDynamics& getFrameDynamics() const { return *frameDynamicsPtr_; }
    virtual const pinocchio::SE3 getTargetPose(ocs2::scalar_t time) const { return targetPose_; }
    virtual const ocs2::vector_t getTargetTwist(ocs2::scalar_t time) const { return targetTwist_; }
    virtual const ocs2::vector_t getTargetAcc(ocs2::scalar_t time) const { return targetAcc_; }

    size_t getNumConstraints(ocs2::scalar_t time) const override { return getConfiguredNumConstraints(); }
    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation& preComp) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::vector_t& input,
                                                                   const ocs2::PreComputation& preComp) const override;

  protected:
    size_t getConfiguredNumConstraints() const;

    std::unique_ptr<ocp_solver::PinocchioFrameDynamics> frameDynamicsPtr_;
    const size_t numConstraints_;
    Config config_;
    pinocchio::SE3 targetPose_ = pinocchio::SE3::Identity();
    ocs2::vector_t targetTwist_ = ocs2::vector_t::Zero(6);
    ocs2::vector_t targetAcc_ = ocs2::vector_t::Zero(6);
  };
}
