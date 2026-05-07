#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include "ocp_solver/quadratic_state_input_cost.h"

namespace ocp_solver {
  QuadraticStateInputCost::QuadraticStateInputCost(const ocs2::PinocchioInterface& pinocchioInterface, ocs2::matrix_t Q, ocs2::matrix_t R, ocs2::matrix_t P)
    : ocs2::QuadraticStateInputCost(Q, R, P),
      pinocchioInterface_(pinocchioInterface) {};

  std::pair<ocs2::vector_t, ocs2::vector_t> QuadraticStateInputCost::getStateInputDeviation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::TargetTrajectories& targetTrajectories) const {
    ocs2::vector_t stateDerivation(2*pinocchioInterface_.getModel().nv);
    stateDerivation.head(pinocchioInterface_.getModel().nv) = pinocchio::difference(pinocchioInterface_.getModel(), targetTrajectories.getDesiredState(time).head(pinocchioInterface_.getModel().nq), state.head(pinocchioInterface_.getModel().nq));
    stateDerivation.tail(pinocchioInterface_.getModel().nv) = state.tail(pinocchioInterface_.getModel().nq) - targetTrajectories.getDesiredState(time).tail(pinocchioInterface_.getModel().nq);
    const ocs2::vector_t inputDeviation = input - targetTrajectories.getDesiredInput(time);
    return {stateDerivation, inputDeviation};
  }
}
