#pragma once

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_core/Types.h>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

namespace ocp_constraint {
  class PositionConstraint final : public ocs2::StateConstraint {
  public:
    PositionConstraint(const pinocchio::FrameIndex targetFrameId,
                       const pinocchio::SE3& targetPose);

    ~PositionConstraint() override = default;

    PositionConstraint* clone() const override;

    bool isActive(ocs2::scalar_t time) const override;

    size_t getNumConstraints(ocs2::scalar_t time) const override { return 6; };

    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::PreComputation& preComp) const override;

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::PreComputation& preComp) const override;

  private:
    pinocchio::FrameIndex targetFrameId_;
    pinocchio::SE3 targetPose_;
    bool isActive_ = true;
  };
}
