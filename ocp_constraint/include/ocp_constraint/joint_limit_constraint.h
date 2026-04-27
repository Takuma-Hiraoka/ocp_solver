#pragma once

#include <memory>

#include <ocs2_core/cost/StateCost.h>
#include "ocp_constraint/penalties/piece_wise_polynominal_barrier_penalty.h"

#include <ocp_solver/state_converter.h>
#include <ocp_solver/switched_model_reference_manager.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

namespace ocp_constraint {
  class JointLimitsConstraint final : public ocs2::StateCost {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      JointLimitsConstraint(const ocs2::PinocchioInterface pinocchioInterface,
                            const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                            ocs2::PieceWisePolynomialBarrierPenalty::Config barrierConfig=ocs2::PieceWisePolynomialBarrierPenalty::Config(1200, 0.1));

    JointLimitsConstraint* clone() const override { return new JointLimitsConstraint(*this); }

    ocs2::scalar_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::TargetTrajectories& targetTrajectories,
                            const ocs2::PreComputation& preComp) const override;

    ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(ocs2::scalar_t time,
                                                                         const ocs2::vector_t& state,
                                                                         const ocs2::TargetTrajectories& targetTrajectories,
                                                                         const ocs2::PreComputation& preComp) const override;

    ocs2::scalar_t getValue(const ocs2::vector_t& jointPositions) const;
    ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(const ocs2::vector_t& jointPositions) const;

    void setGains(const ocs2::scalar_t& mu, const ocs2::scalar_t& delta);
    void getGains(ocs2::scalar_t& mu, ocs2::scalar_t& delta) const;

  private:
    JointLimitsConstraint(const JointLimitsConstraint& rhs);

    std::unique_ptr<ocs2::PieceWisePolynomialBarrierPenalty> penaltyPtr_;
    const ocp_solver::StateConverter<ocs2::scalar_t>* stateConverterPtr_;
    std::pair<ocs2::vector_t, ocs2::vector_t> positionLimits_;
  };

}
