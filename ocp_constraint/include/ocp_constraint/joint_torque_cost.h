#pragma once

#include <ocs2_core/cost/StateInputCost.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocp_solver/solver/state_converter.h>

namespace ocp_constraint {
  class JointTorqueCost : public ocs2::StateInputCost {
  public:
    JointTorqueCost(const ocs2::matrix_t& weights, const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter);
    ~JointTorqueCost() override = default;
    JointTorqueCost* clone() const override { return new JointTorqueCost(*this); }

    ocs2::scalar_t getValue(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::TargetTrajectories& targetTrajectories,
                            const ocs2::PreComputation& preComp) const final;

    ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input,
                                                                         const ocs2::TargetTrajectories& targetTrajectories,
                                                                         const ocs2::PreComputation& preComp) const final;

  private:
   ocs2::matrix_t sqrtWeights_;
   ocp_solver::StateConverter<ocs2::scalar_t>* stateConverter_;
  };
}

