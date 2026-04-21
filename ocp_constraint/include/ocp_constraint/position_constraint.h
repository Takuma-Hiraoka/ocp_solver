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
    PositionConstraint(const ocs2::PinocchioInterface& pinocchioInterface,
                       const std::string& targetFrameName,
                       const Eigen::Vector3d& targetPosition);

    ~PositionConstraint() override = default;

    PositionConstraint* clone() const override;

    size_t getNumConstraints(ocs2::scalar_t time) const override { return 3; };

    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::PreComputation& preComp) const override;

    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::PreComputation& preComp) const override;

  private:
    mutable ocs2::PinocchioInterface pinocchioInterface_;

    pinocchio::FrameIndex targetFrameId_;
    Eigen::Vector3d targetPosition_;
  };
}
