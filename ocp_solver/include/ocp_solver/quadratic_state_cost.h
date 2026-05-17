#pragma once

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

namespace ocp_solver {
  class QuadraticStateCost : public ocs2::QuadraticStateCost {
  public:
    QuadraticStateCost(const ocs2::PinocchioInterface& pinocchioInterface, ocs2::matrix_t Q);
  protected:
    ocs2::vector_t getStateDeviation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::TargetTrajectories& targetTrajectories) const override;
  private:
    const pinocchio::Model model_;
  };
}
