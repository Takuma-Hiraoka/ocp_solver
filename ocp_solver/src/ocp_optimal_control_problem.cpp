#include "ocp_solver/ocp_optimal_control_problem.h"
#include "ocp_solver/ocp_state_cost_collection.h"
#include "ocp_solver/ocp_state_input_cost_collection.h"
#include "ocp_solver/ocp_state_constraint_collection.h"
#include "ocp_solver/ocp_state_input_constraint_collection.h"

namespace ocp_solver {
  OptimalControlProblem::OptimalControlProblem(const size_t& ss, const size_t& is)
    : ocs2::OptimalControlProblem() {
    /* Cost */
    costPtr.reset(new StateInputCostCollection(ss, is));
    stateCostPtr.reset(new StateCostCollection(ss));
    preJumpCostPtr.reset(new StateCostCollection(ss));
    finalCostPtr.reset(new StateCostCollection(ss));
    /* Soft constraints */
    softConstraintPtr.reset(new StateInputCostCollection(ss, is));
    stateSoftConstraintPtr.reset(new StateCostCollection(ss));
    preJumpSoftConstraintPtr.reset(new StateCostCollection(ss));
    finalSoftConstraintPtr.reset(new StateCostCollection(ss));
    /* Equality constraints */
    equalityConstraintPtr.reset(new StateInputConstraintCollection(ss, is));
    stateEqualityConstraintPtr.reset(new StateConstraintCollection(ss));
    preJumpEqualityConstraintPtr.reset(new StateConstraintCollection(ss));
    finalEqualityConstraintPtr.reset(new StateConstraintCollection(ss));
    /* Inequality constraints */
    inequalityConstraintPtr.reset(new StateInputConstraintCollection(ss, is));
    stateInequalityConstraintPtr.reset(new StateConstraintCollection(ss));
    preJumpInequalityConstraintPtr.reset(new StateConstraintCollection(ss));
    finalInequalityConstraintPtr.reset(new StateConstraintCollection(ss));
  }

}
