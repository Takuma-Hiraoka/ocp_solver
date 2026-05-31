#include <ocs2_core/misc/LinearAlgebra.h>

#include "ocs2_oc/approximate_model/ChangeOfInputVariables.h"
#include "ocs2_oc/approximate_model/LinearQuadraticApproximator.h"

#include "ocp_solver/solver/ocp_transcription.h"
#include "ocp_solver/solver/ocp_pre_computation.h"

namespace ocp_solver {
  ocs2::multiple_shooting::Transcription setupIntermediateNode(ocs2::OptimalControlProblem& optimalControlProblem, ocs2::DynamicsSensitivityDiscretizer& sensitivityDiscretizer,
                                                               ocs2::scalar_t t, ocs2::scalar_t dt, const ocs2::vector_t& x, const ocs2::vector_t& x_next, const ocs2::vector_t& u) {
    // Results and short-hand notation
    ocs2::multiple_shooting::Transcription transcription;
    ocs2::ScalarFunctionQuadraticApproximation& cost = transcription.cost;
    ocs2::VectorFunctionLinearApproximation& dynamics = transcription.dynamics;
    ocs2::multiple_shooting::ConstraintsSize& constraintsSize = transcription.constraintsSize;
    ocs2::VectorFunctionLinearApproximation& stateEqConstraints = transcription.stateEqConstraints;
    ocs2::VectorFunctionLinearApproximation& stateInputEqConstraints = transcription.stateInputEqConstraints;
    ocs2::VectorFunctionLinearApproximation& stateIneqConstraints = transcription.stateIneqConstraints;
    ocs2::VectorFunctionLinearApproximation& stateInputIneqConstraints = transcription.stateInputIneqConstraints;

    // Dynamics
    // Discretization returns x_{k+1} = A_{k} * dx_{k} + B_{k} * du_{k} + b_{k}
    dynamics = sensitivityDiscretizer(*optimalControlProblem.dynamicsPtr, t, x, u, dt);

    ocs2::PinocchioInterface& pinocchioInterface = static_cast<const ocp_solver::OCPPreComputation&>(*optimalControlProblem.preComputationPtr).getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    ocs2::vector_t x_integrated = dynamics.f;
    ocs2::vector_t x_diff(dynamics.f.size() - (model.nq - model.nv));
    x_diff.head(model.nv) = pinocchio::difference(model, x_next.head(model.nq), x_integrated.head(model.nq));
    x_diff.segment(model.nv, model.nv) = x_integrated.segment(model.nq, model.nv) - x_next.segment(model.nq, model.nv);
    if (x_integrated.size() > model.nq + model.nv) {
      const Eigen::Index extraStart = model.nq + model.nv;
      x_diff.tail(x_integrated.size() - extraStart) =
        x_integrated.tail(x_integrated.size() - extraStart) - x_next.tail(x_integrated.size() - extraStart);
    }
    dynamics.f = x_diff;

    // Precomputation for other terms
    constexpr ocs2::RequestSet request = ocs2::Request::Cost + ocs2::Request::SoftConstraint + ocs2::Request::Constraint + ocs2::Request::Approximation;
    optimalControlProblem.preComputationPtr->request(request, t, x, u);

    // Costs: Approximate the integral with forward euler
    cost = approximateCost(optimalControlProblem, t, x, u);
    cost *= dt;

    // State equality constraints
    if (!optimalControlProblem.stateEqualityConstraintPtr->empty()) {
      constraintsSize.stateEq = optimalControlProblem.stateEqualityConstraintPtr->getTermsSize(t);
      stateEqConstraints =
        optimalControlProblem.stateEqualityConstraintPtr->getLinearApproximation(t, x, *optimalControlProblem.preComputationPtr);
    }

    // State-input equality constraints
    if (!optimalControlProblem.equalityConstraintPtr->empty()) {
      constraintsSize.stateInputEq = optimalControlProblem.equalityConstraintPtr->getTermsSize(t);
      stateInputEqConstraints =
        optimalControlProblem.equalityConstraintPtr->getLinearApproximation(t, x, u, *optimalControlProblem.preComputationPtr);
    }

    // State inequality constraints.
    if (!optimalControlProblem.stateInequalityConstraintPtr->empty()) {
      constraintsSize.stateIneq = optimalControlProblem.stateInequalityConstraintPtr->getTermsSize(t);
      stateIneqConstraints =
        optimalControlProblem.stateInequalityConstraintPtr->getLinearApproximation(t, x, *optimalControlProblem.preComputationPtr);
    }

    // State-input inequality constraints.
    if (!optimalControlProblem.inequalityConstraintPtr->empty()) {
      constraintsSize.stateInputIneq = optimalControlProblem.inequalityConstraintPtr->getTermsSize(t);
      stateInputIneqConstraints =
        optimalControlProblem.inequalityConstraintPtr->getLinearApproximation(t, x, u, *optimalControlProblem.preComputationPtr);
    }

    return transcription;

  }

  ocs2::multiple_shooting::EventTranscription setupEventNode(ocs2::OptimalControlProblem& optimalControlProblem, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& x_next) {
    // Results and short-hand notation
    ocs2::multiple_shooting::EventTranscription transcription;
    ocs2::ScalarFunctionQuadraticApproximation& cost = transcription.cost;
    ocs2::VectorFunctionLinearApproximation& dynamics = transcription.dynamics;
    ocs2::multiple_shooting::ConstraintsSize& constraintsSize = transcription.constraintsSize;
    ocs2::VectorFunctionLinearApproximation& eqConstraints = transcription.eqConstraints;
    ocs2::VectorFunctionLinearApproximation& ineqConstraints = transcription.ineqConstraints;

    // Dynamics
    // jump map returns // x_{k+1} = A_{k} * dx_{k} + b_{k}
    dynamics = optimalControlProblem.dynamicsPtr->jumpMapLinearApproximation(t, x);

    ocs2::PinocchioInterface& pinocchioInterface = static_cast<const ocp_solver::OCPPreComputation&>(*optimalControlProblem.preComputationPtr).getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    ocs2::vector_t x_integrated = dynamics.f;
    ocs2::vector_t x_diff(dynamics.f.size() - (model.nq - model.nv));
    x_diff.head(model.nv) = pinocchio::difference(model, x_next.head(model.nq), x_integrated.head(model.nq));
    x_diff.segment(model.nv, model.nv) = x_integrated.segment(model.nq, model.nv) - x_next.segment(model.nq, model.nv);
    if (x_integrated.size() > model.nq + model.nv) {
      const Eigen::Index extraStart = model.nq + model.nv;
      x_diff.tail(x_integrated.size() - extraStart) =
        x_integrated.tail(x_integrated.size() - extraStart) - x_next.tail(x_integrated.size() - extraStart);
    }
    dynamics.f = x_diff;

    dynamics.dfdu.setZero(dynamics.f.size(), 0);  // Overwrite derivative that shouldn't exist.

    constexpr ocs2::RequestSet request = ocs2::Request::Cost + ocs2::Request::SoftConstraint + ocs2::Request::Dynamics + ocs2::Request::Approximation;
    optimalControlProblem.preComputationPtr->requestPreJump(request, t, x);

    // Costs
    cost = approximateEventCost(optimalControlProblem, t, x);

    // State equality constraints.
    if (!optimalControlProblem.preJumpEqualityConstraintPtr->empty()) {
      constraintsSize.stateEq = optimalControlProblem.preJumpEqualityConstraintPtr->getTermsSize(t);
      eqConstraints =
        optimalControlProblem.preJumpEqualityConstraintPtr->getLinearApproximation(t, x, *optimalControlProblem.preComputationPtr);
    }

    // State inequality constraints.
    if (!optimalControlProblem.preJumpInequalityConstraintPtr->empty()) {
      constraintsSize.stateIneq = optimalControlProblem.preJumpInequalityConstraintPtr->getTermsSize(t);
      ineqConstraints =
        optimalControlProblem.preJumpInequalityConstraintPtr->getLinearApproximation(t, x, *optimalControlProblem.preComputationPtr);
    }

    return transcription;

  }
}
