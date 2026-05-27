#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include "ocp_solver/common/quadratic_state_cost.h"

namespace ocp_solver {
  QuadraticStateCost::QuadraticStateCost(const ocs2::PinocchioInterface& pinocchioInterface, ocs2::matrix_t Q)
    : ocs2::QuadraticStateCost(Q),
      model_(pinocchioInterface.getModel()) {};

  ocs2::vector_t QuadraticStateCost::getStateDeviation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::TargetTrajectories& targetTrajectories) const {
    ocs2::vector_t stateDerivation(2*model_.nv);
    stateDerivation.head(model_.nv) = pinocchio::difference(model_, targetTrajectories.getDesiredState(time).head(model_.nq), state.head(model_.nq));
    stateDerivation.segment(model_.nv, model_.nv) =
      state.segment(model_.nq, model_.nv) - targetTrajectories.getDesiredState(time).segment(model_.nq, model_.nv);
    return stateDerivation;
  }
}
