#pragma once

#include <memory>

#include <ocp_solver/pinocchio/pinocchio_frame_dynamics.h>
#include <ocp_solver/solver/trajectory.h>
#include <ocs2_core/constraint/StateConstraint.h>

namespace ocp_constraint {

  class PointConstraint final : public ocs2::StateConstraint {
  public:
    PointConstraint(const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                    ocp_solver::TargetSE3Trajectory targetTrajectory);
    ~PointConstraint() override = default;
    PointConstraint* clone() const override { return new PointConstraint(*this); }

    size_t getNumConstraints(ocs2::scalar_t time) const override { return 6; };
    ocs2::vector_t getValue(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::PreComputation& preComputation) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::PreComputation& preComputation) const override;

  private:
    PointConstraint(const PointConstraint& rhs);
    ocp_solver::TargetSE3Trajectory targetTrajectory_;
    std::unique_ptr<ocp_solver::PinocchioFrameDynamics> frameDynamicsPtr_;
  };

}
