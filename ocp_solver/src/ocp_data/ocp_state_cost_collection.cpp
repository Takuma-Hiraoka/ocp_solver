#include "ocp_solver/ocp_data/ocp_state_cost_collection.h"

namespace ocp_solver {

  StateCostCollection::StateCostCollection(const size_t& ss)
    : state_variable_dim_(ss) {}

  StateCostCollection* StateCostCollection::clone() const {
    return new StateCostCollection(*this);
  }
  ocs2::ScalarFunctionQuadraticApproximation StateCostCollection::getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                                            const ocs2::TargetTrajectories& targetTrajectories,
                                                                                            const ocs2::PreComputation& preComp) const {
    std::vector<std::unique_ptr<ocs2::StateCost>>::const_iterator firstActive =
      std::find_if(terms_.begin(), terms_.end(), [time](const std::unique_ptr<ocs2::StateCost>& costTerm) { return costTerm->isActive(time); });

    // No active terms (or terms is empty).
    if (firstActive == terms_.end()) {
      return ocs2::ScalarFunctionQuadraticApproximation::Zero(state_variable_dim_);
    }

    // Initialize with first active term, accumulate potentially other active terms.
    ocs2::ScalarFunctionQuadraticApproximation cost = (*firstActive)->getQuadraticApproximation(time, state, targetTrajectories, preComp);
    std::for_each(std::next(firstActive), terms_.end(), [&](const std::unique_ptr<ocs2::StateCost>& costTerm) {
                                                          if (costTerm->isActive(time)) {
                                                            const ocs2::ScalarFunctionQuadraticApproximation costTermApproximation = costTerm->getQuadraticApproximation(time, state, targetTrajectories, preComp);
                                                            cost.f += costTermApproximation.f;
                                                            cost.dfdx += costTermApproximation.dfdx;
                                                            cost.dfdxx += costTermApproximation.dfdxx;
                                                          }
                                                        });

    // Make sure that input derivatives are empty
    cost.dfdu = ocs2::vector_t();
    cost.dfduu = ocs2::matrix_t();
    cost.dfdux = ocs2::matrix_t();

    return cost;
  }

}
