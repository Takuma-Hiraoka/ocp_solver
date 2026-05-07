#include "ocp_solver/ocp_pre_computation.h"
#include "ocp_solver/ocp_metrics_computation.h"

#include <ocs2_oc/approximate_model/LinearQuadraticApproximator.h>

namespace ocp_solver {

  ocs2::Metrics computeIntermediateMetrics(ocs2::OptimalControlProblem& optimalControlProblem, ocs2::DynamicsDiscretizer& discretizer, ocs2::scalar_t t, ocs2::scalar_t dt,
                                     const ocs2::vector_t& x, const ocs2::vector_t& x_next, const ocs2::vector_t& u) {
    // Dynamics
    ocs2::PinocchioInterface& pinocchioInterface = static_cast<const ocp_solver::OCPPreComputation&>(*optimalControlProblem.preComputationPtr).getPinocchioInterface();
    ocs2::vector_t x_integrated = discretizer(*optimalControlProblem.dynamicsPtr, t, x, u, dt);
    ocs2::vector_t x_diff(2*pinocchioInterface.getModel().nv);
    x_diff.head(pinocchioInterface.getModel().nv) = pinocchio::difference(pinocchioInterface.getModel(), x_next.head(pinocchioInterface.getModel().nq), x_integrated.head(pinocchioInterface.getModel().nq));
    x_diff.tail(pinocchioInterface.getModel().nv) = x_integrated.tail(pinocchioInterface.getModel().nv) - x_next.tail(pinocchioInterface.getModel().nv);

    // Precomputation
    constexpr auto request = ocs2::Request::Cost + ocs2::Request::SoftConstraint + ocs2::Request::Constraint;
    optimalControlProblem.preComputationPtr->request(request, t, x, u);

    // Compute metrics
    auto metrics = ocs2::computeIntermediateMetrics(optimalControlProblem, t, x, u, std::move(x_diff));
    metrics.cost *= dt;  // consider dt

    return metrics;
  }

  ocs2::Metrics computeTerminalMetrics(ocs2::OptimalControlProblem& optimalControlProblem, ocs2::scalar_t t, const ocs2::vector_t& x) {
    // Precomputation
    constexpr auto request = ocs2::Request::Cost + ocs2::Request::SoftConstraint + ocs2::Request::Constraint;
    optimalControlProblem.preComputationPtr->requestFinal(request, t, x);

    return ocs2::computeFinalMetrics(optimalControlProblem, t, x);
  }

  ocs2::Metrics computeEventMetrics(ocs2::OptimalControlProblem& optimalControlProblem, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& x_next) {
    // Precomputation
    constexpr auto request = ocs2::Request::Cost + ocs2::Request::SoftConstraint + ocs2::Request::Constraint + ocs2::Request::Dynamics;
    optimalControlProblem.preComputationPtr->requestPreJump(request, t, x);

    // Dynamics
    ocs2::PinocchioInterface& pinocchioInterface = static_cast<const ocp_solver::OCPPreComputation&>(*optimalControlProblem.preComputationPtr).getPinocchioInterface();
    ocs2::vector_t x_integrated = optimalControlProblem.dynamicsPtr->computeJumpMap(t, x);
    ocs2::vector_t x_diff(2*pinocchioInterface.getModel().nv);
    x_diff.head(pinocchioInterface.getModel().nv) = pinocchio::difference(pinocchioInterface.getModel(), x_next.head(pinocchioInterface.getModel().nq), x_integrated.head(pinocchioInterface.getModel().nq));
    x_diff.tail(pinocchioInterface.getModel().nv) = x_integrated.tail(pinocchioInterface.getModel().nv) - x_next.tail(pinocchioInterface.getModel().nv);

    return ocs2::computePreJumpMetrics(optimalControlProblem, t, x, std::move(x_diff));
  }

}
