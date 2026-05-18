#pragma once

#include <memory>

#include <ocp_solver/pinocchio/pinocchio_frame_dynamics.h>
#include <ocs2_core/constraint/StateConstraint.h>

namespace ocp_constraint {

  class PointConstraint final : public ocs2::StateConstraint {
  public:
    PointConstraint(const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                    const pinocchio::SE3 targetPose);
    ~PointConstraint() override = default;
    PointConstraint* clone() const override { return new PointConstraint(*this); }

    size_t getNumConstraints(ocs2::scalar_t time) const override { return 6; };
    ocs2::vector_t getValue(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::PreComputation& preComputation) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::PreComputation& preComputation) const override;

  private:
    PointConstraint(const PointConstraint& rhs);
    pinocchio::SE3 targetPose_;
    std::unique_ptr<ocp_solver::PinocchioFrameDynamics> frameDynamicsPtr_;
  };

}
