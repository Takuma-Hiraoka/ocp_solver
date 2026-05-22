#include "ocp_solver/ocp_data/ocp_state_input_cost_collection.h"

namespace ocp_solver {

  StateInputCostCollection::StateInputCostCollection(const size_t& ss, const size_t& is)
    : state_variable_dim_(ss),
      input_variable_dim_(is) {}

  StateInputCostCollection* StateInputCostCollection::clone() const {
    return new StateInputCostCollection(*this);
  }

  ocs2::ScalarFunctionQuadraticApproximation StateInputCostCollection::getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                                                 const ocs2::vector_t& input,
                                                                                                 const ocs2::TargetTrajectories& targetTrajectories,
                                                                                                 const ocs2::PreComputation& preComp) const {
    std::vector<std::unique_ptr<ocs2::StateInputCost>>::const_iterator firstActive = std::find_if(terms_.begin(), terms_.end(),
                                          [time](const std::unique_ptr<ocs2::StateInputCost>& costTerm) { return costTerm->isActive(time); });

    // No active terms (or terms is empty).
    if (firstActive == terms_.end()) {
      return ocs2::ScalarFunctionQuadraticApproximation::Zero(state_variable_dim_, input_variable_dim_);
    }

    // Initialize with first active term, accumulate potentially other active terms.
    ocs2::ScalarFunctionQuadraticApproximation cost = (*firstActive)->getQuadraticApproximation(time, state, input, targetTrajectories, preComp);
    std::for_each(std::next(firstActive), terms_.end(), [&](const std::unique_ptr<ocs2::StateInputCost>& costTerm) {
                                                          if (costTerm->isActive(time)) {
                                                            cost += costTerm->getQuadraticApproximation(time, state, input, targetTrajectories, preComp);
                                                          }
                                                        });

    return cost;
  }

}
