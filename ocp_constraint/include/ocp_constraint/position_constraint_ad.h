#pragma once

#include <memory>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/pinocchio/pinocchio_frame_dynamics_cppad.h>

namespace ocp_constraint {

  class PositionConstraintAD : public ocs2::StateInputConstraint {
  public:
    struct Config {
      ocs2::matrix_t Ax;
      ocs2::matrix_t Av;
      ocs2::matrix_t Aa;
    };

    PositionConstraintAD(const ocp_solver::PinocchioFrameDynamicsCppAd& frameDynamics,
                         size_t numConstraints,
                         Config config = Config(),
                         pinocchio::SE3 targetPose = pinocchio::SE3::Identity(),
                         ocs2::vector_t targetTwist = ocs2::vector_t::Zero(6),
                         ocs2::vector_t targetAcc = ocs2::vector_t::Zero(6));

    ~PositionConstraintAD() override = default;
    PositionConstraintAD* clone() const override { return new PositionConstraintAD(*this); }

    void configure(Config&& config);
    void configure(const Config& config) { this->configure(Config(config)); }

    const ocp_solver::PinocchioFrameDynamicsCppAd& getFrameDynamics() const { return *frameDynamicsPtr_; }
    virtual const pinocchio::SE3 getTargetPose(ocs2::scalar_t time) const { return targetPose_; }
    virtual const ocs2::vector_t getTargetTwist(ocs2::scalar_t time) const { return targetTwist_; }
    virtual const ocs2::vector_t getTargetAcc(ocs2::scalar_t time) const { return targetAcc_; }

    size_t getNumConstraints(ocs2::scalar_t time) const override { return numConstraints_; }
    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation& preComp) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::vector_t& input,
                                                                   const ocs2::PreComputation& preComp) const override;

  protected:
    PositionConstraintAD(const PositionConstraintAD& rhs);

  private:
    std::unique_ptr<ocp_solver::PinocchioFrameDynamicsCppAd> frameDynamicsPtr_;
    const size_t numConstraints_;
    Config config_;
    pinocchio::SE3 targetPose_ = pinocchio::SE3::Identity();
    ocs2::vector_t targetTwist_ = ocs2::vector_t::Zero(6);
    ocs2::vector_t targetAcc_ = ocs2::vector_t::Zero(6);
  };

}
