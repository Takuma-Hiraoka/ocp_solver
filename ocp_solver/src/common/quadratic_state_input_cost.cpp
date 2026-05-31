#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include "ocp_solver/common/quadratic_state_input_cost.h"

namespace ocp_solver {
  QuadraticStateInputCost::QuadraticStateInputCost(const ocs2::PinocchioInterface& pinocchioInterface, ocs2::matrix_t Q, ocs2::matrix_t R, ocs2::matrix_t P)
    : model_(pinocchioInterface.getModel()),
      Q_(std::move(Q)), R_(std::move(R)), P_(std::move(P)) {
    if (P_.size() > 0) {
      assert(P_.rows() == R_.rows());
      assert(P_.cols() == Q_.rows());
    }

  };

  ocs2::scalar_t QuadraticStateInputCost::getValue(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input,
                                                   const ocs2::TargetTrajectories& targetTrajectories, const ocs2::PreComputation&) const {
    ocs2::vector_t stateDeviation, inputDeviation;
    std::tie(stateDeviation, inputDeviation) = getStateInputDeviation(time, state, input, targetTrajectories);

    if (P_.size() == 0) {
      return 0.5 * stateDeviation.dot(Q_ * stateDeviation) + 0.5 * inputDeviation.dot(R_ * inputDeviation);
    } else {
      return 0.5 * stateDeviation.dot(Q_ * stateDeviation) + 0.5 * inputDeviation.dot(R_ * inputDeviation) +
        inputDeviation.dot(P_ * stateDeviation);
    }
  }

  ocs2::ScalarFunctionQuadraticApproximation QuadraticStateInputCost::getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                                                const ocs2::vector_t& input,
                                                                                                const ocs2::TargetTrajectories& targetTrajectories,
                                                                                                const ocs2::PreComputation&) const {
    ocs2::vector_t stateDeviation, inputDeviation;
    std::tie(stateDeviation, inputDeviation) = getStateInputDeviation(time, state, input, targetTrajectories);

    ocs2::ScalarFunctionQuadraticApproximation L;
    L.dfdxx = Q_;
    L.dfduu = R_;
    L.dfdx.noalias() = Q_ * stateDeviation;
    L.dfdu.noalias() = R_ * inputDeviation;
    L.f = 0.5 * stateDeviation.dot(L.dfdx) + 0.5 * inputDeviation.dot(L.dfdu);

    if (P_.size() == 0) {
      L.dfdux.setZero(input.size(), stateDeviation.size());

    } else {
      const ocs2::vector_t pDeviation = P_ * stateDeviation;
      L.f += inputDeviation.dot(pDeviation);
      L.dfdu += pDeviation;
      L.dfdx.noalias() += P_.transpose() * inputDeviation;
      L.dfdux = P_;
    }

    return L;
  }

  std::pair<ocs2::vector_t, ocs2::vector_t> QuadraticStateInputCost::getStateInputDeviation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::TargetTrajectories& targetTrajectories) const {
    const ocs2::vector_t& targetState = targetTrajectories.getDesiredState(time);
    ocs2::vector_t stateDerivation(state.size() - (model_.nq - model_.nv));
    stateDerivation.head(model_.nv) = pinocchio::difference(model_, targetTrajectories.getDesiredState(time).head(model_.nq), state.head(model_.nq));
    stateDerivation.segment(model_.nv, model_.nv) =
      state.segment(model_.nq, model_.nv) - targetTrajectories.getDesiredState(time).segment(model_.nq, model_.nv);
    if (state.size() > model_.nq + model_.nv) {
      const Eigen::Index extraStart = model_.nq + model_.nv;
      stateDerivation.tail(state.size() - extraStart) = state.tail(state.size() - extraStart) - targetState.tail(state.size() - extraStart);
    }
    const ocs2::vector_t inputDeviation = input - targetTrajectories.getDesiredInput(time);
    return {stateDerivation, inputDeviation};
  }
}
