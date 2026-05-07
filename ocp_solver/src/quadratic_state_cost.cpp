#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include "ocp_solver/quadratic_state_cost.h"

namespace ocp_solver {
  QuadraticStateCost::QuadraticStateCost(const ocs2::PinocchioInterface& pinocchioInterface, ocs2::matrix_t Q)
    : ocs2::QuadraticStateCost(Q),
      pinocchioInterface_(pinocchioInterface) {};

  ocs2::vector_t QuadraticStateCost::getStateDeviation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::TargetTrajectories& targetTrajectories) const {
    ocs2::vector_t stateDerivation(2*pinocchioInterface_.getModel().nv);
    stateDerivation.head(pinocchioInterface_.getModel().nv) = pinocchio::difference(pinocchioInterface_.getModel(), targetTrajectories.getDesiredState(time).head(pinocchioInterface_.getModel().nq), state.head(pinocchioInterface_.getModel().nq));
    stateDerivation.tail(pinocchioInterface_.getModel().nv) = state.tail(pinocchioInterface_.getModel().nq) - targetTrajectories.getDesiredState(time).tail(pinocchioInterface_.getModel().nq);
    return stateDerivation;
  }
}
