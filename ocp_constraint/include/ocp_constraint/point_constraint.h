#pragma once

#include <memory>

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_core/constraint/StateConstraint.h>
#include <ocp_solver/ocp_pinocchio_mapping.h>

namespace ocp_constraint {

  class PointConstraint final : public ocs2::StateConstraint {
  public:
    PointConstraint(const ocs2::PinocchioInterface pinocchioInterface,
                    const ocp_solver::OCPPinocchioMapping mapping,
                    const std::string targetFrameName,
                    const pinocchio::SE3 targetPose);
    ~PointConstraint() override = default;
    PointConstraint* clone() const override { return new PointConstraint(pinocchioInterface_, mapping_, targetFrameName_, targetPose_); }

    size_t getNumConstraints(ocs2::scalar_t time) const override { return 6; };
    ocs2::vector_t getValue(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::PreComputation& preComputation) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::PreComputation& preComputation) const override;

  private:
    pinocchio::SE3 targetPose_;
    std::unique_ptr<ocs2::PinocchioEndEffectorKinematics> endEffectorKinematicsPtr_;

    const ocs2::PinocchioInterface pinocchioInterface_;
    const ocp_solver::OCPPinocchioMapping mapping_;
    const std::string targetFrameName_;
  };

}
