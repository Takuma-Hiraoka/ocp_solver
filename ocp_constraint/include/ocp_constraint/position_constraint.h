#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/pinocchio/pinocchio_frame_dynamics.h>
#include <ocp_solver/solver/switched_model_reference_manager.h>
#include <ocp_solver/solver/trajectory.h>

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
                       ocp_solver::TargetSE3Trajectory targetTrajectory = ocp_solver::TargetSE3Trajectory());

    ~PositionConstraint() override = default;
    PositionConstraint* clone() const override { return new PositionConstraint(*this); }
    PositionConstraint(const PositionConstraint& rhs);

    void configure(Config&& config);
    void configure(const Config& config) { this->configure(Config(config)); }

    const ocp_solver::PinocchioFrameDynamics& getFrameDynamics() const { return *frameDynamicsPtr_; }
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
    size_t getConfiguredNumConstraints() const;

    std::unique_ptr<ocp_solver::PinocchioFrameDynamics> frameDynamicsPtr_;
    const size_t numConstraints_;
    Config config_;
    ocp_solver::TargetSE3Trajectory targetTrajectory_;
  };
}
