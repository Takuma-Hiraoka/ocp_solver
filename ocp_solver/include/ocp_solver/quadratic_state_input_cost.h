#pragma once

#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

namespace ocp_solver {
  class QuadraticStateInputCost : public ocs2::QuadraticStateInputCost {
  public:
    QuadraticStateInputCost(const ocs2::PinocchioInterface& pinocchioInterface, ocs2::matrix_t Q, ocs2::matrix_t R, ocs2::matrix_t P = ocs2::matrix_t());
  protected:
    std::pair<ocs2::vector_t, ocs2::vector_t> getStateInputDeviation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::TargetTrajectories& targetTrajectories) const override;
  private:
    const pinocchio::Model model_;
  };
}
