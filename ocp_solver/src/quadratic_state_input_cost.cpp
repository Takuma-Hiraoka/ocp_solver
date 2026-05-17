#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include "ocp_solver/quadratic_state_input_cost.h"

namespace ocp_solver {
  QuadraticStateInputCost::QuadraticStateInputCost(const ocs2::PinocchioInterface& pinocchioInterface, ocs2::matrix_t Q, ocs2::matrix_t R, ocs2::matrix_t P)
    : ocs2::QuadraticStateInputCost(Q, R, P),
      model_(pinocchioInterface.getModel()) {};

  std::pair<ocs2::vector_t, ocs2::vector_t> QuadraticStateInputCost::getStateInputDeviation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::TargetTrajectories& targetTrajectories) const {
    ocs2::vector_t stateDerivation(2*model_.nv);
    stateDerivation.head(model_.nv) = pinocchio::difference(model_, targetTrajectories.getDesiredState(time).head(model_.nq), state.head(model_.nq));
    stateDerivation.tail(model_.nv) = state.tail(model_.nq) - targetTrajectories.getDesiredState(time).tail(model_.nq);
    const ocs2::vector_t inputDeviation = input - targetTrajectories.getDesiredInput(time);
    return {stateDerivation, inputDeviation};
  }
}
