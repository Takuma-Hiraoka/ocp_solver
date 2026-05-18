#pragma once

#include <ocs2_core/cost/StateInputCost.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

namespace ocp_solver {
  class QuadraticStateInputCost : public ocs2::StateInputCost {
  public:
    QuadraticStateInputCost(const ocs2::PinocchioInterface& pinocchioInterface, ocs2::matrix_t Q, ocs2::matrix_t R, ocs2::matrix_t P = ocs2::matrix_t());
      ~QuadraticStateInputCost() override = default;
    QuadraticStateInputCost* clone() const override { return new QuadraticStateInputCost(*this); }

    ocs2::scalar_t getValue(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::TargetTrajectories& targetTrajectories,
                      const ocs2::PreComputation&) const final;

    ocs2::ScalarFunctionQuadraticApproximation getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input,
                                                                   const ocs2::TargetTrajectories& targetTrajectories,
                                                                   const ocs2::PreComputation&) const final;


  protected:
      QuadraticStateInputCost(const QuadraticStateInputCost& rhs) = default;

    std::pair<ocs2::vector_t, ocs2::vector_t> getStateInputDeviation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::TargetTrajectories& targetTrajectories) const;
  private:
    const pinocchio::Model model_;
    ocs2::matrix_t Q_;
    ocs2::matrix_t R_;
    ocs2::matrix_t P_;
  };
}
