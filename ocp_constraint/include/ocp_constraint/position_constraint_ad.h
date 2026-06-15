#pragma once

#include <memory>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/pinocchio/pinocchio_frame_dynamics_cppad.h>
#include <ocp_solver/solver/trajectory.h>

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
                         ocp_solver::TargetSE3Trajectory targetTrajectory = ocp_solver::TargetSE3Trajectory());

    ~PositionConstraintAD() override = default;
    PositionConstraintAD* clone() const override { return new PositionConstraintAD(*this); }

    void configure(Config&& config);
    void configure(const Config& config) { this->configure(Config(config)); }

    const ocp_solver::PinocchioFrameDynamicsCppAd& getFrameDynamics() const { return *frameDynamicsPtr_; }
    virtual const pinocchio::SE3 getTargetPose(ocs2::scalar_t time) const { return targetTrajectory_.getTargetPose(time); }
    virtual const ocs2::vector_t getTargetTwist(ocs2::scalar_t time) const { return targetTrajectory_.getTargetTwist(time); }
    virtual const ocs2::vector_t getTargetAcc(ocs2::scalar_t time) const { return targetTrajectory_.getTargetAcc(time); }

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
    PositionConstraintAD(const PositionConstraintAD& rhs);
    size_t getConfiguredNumConstraints() const;

  private:
    std::unique_ptr<ocp_solver::PinocchioFrameDynamicsCppAd> frameDynamicsPtr_;
    const size_t numConstraints_;
    Config config_;
    ocp_solver::TargetSE3Trajectory targetTrajectory_;
  };

}
