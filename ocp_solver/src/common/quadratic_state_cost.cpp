#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include "ocp_solver/common/quadratic_state_cost.h"

namespace ocp_solver {
  QuadraticStateCost::QuadraticStateCost(const ocs2::PinocchioInterface& pinocchioInterface, ocs2::matrix_t Q)
    : ocs2::QuadraticStateCost(Q),
      model_(pinocchioInterface.getModel()) {};

  ocs2::vector_t QuadraticStateCost::getStateDeviation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::TargetTrajectories& targetTrajectories) const {
    const ocs2::vector_t& targetState = targetTrajectories.getDesiredState(time);
    ocs2::vector_t stateDerivation(state.size() - (model_.nq - model_.nv));
    stateDerivation.head(model_.nv) = pinocchio::difference(model_, targetTrajectories.getDesiredState(time).head(model_.nq), state.head(model_.nq));
    stateDerivation.segment(model_.nv, model_.nv) =
      state.segment(model_.nq, model_.nv) - targetTrajectories.getDesiredState(time).segment(model_.nq, model_.nv);
    if (state.size() > model_.nq + model_.nv) {
      const Eigen::Index extraStart = model_.nq + model_.nv;
      stateDerivation.tail(state.size() - extraStart) = state.tail(state.size() - extraStart) - targetState.tail(state.size() - extraStart);
    }
    return stateDerivation;
  }
}
