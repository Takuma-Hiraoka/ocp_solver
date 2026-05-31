#include "ocp_solver/ocp_data/ocp_state_cost_collection.h"

#include <algorithm>

namespace ocp_solver {
  namespace {
    void addPadded(ocs2::ScalarFunctionQuadraticApproximation& total,
                   const ocs2::ScalarFunctionQuadraticApproximation& term) {
      total.f += term.f;
      total.dfdx.head(std::min(total.dfdx.size(), term.dfdx.size())).noalias() +=
        term.dfdx.head(std::min(total.dfdx.size(), term.dfdx.size()));
      const Eigen::Index rows = std::min(total.dfdxx.rows(), term.dfdxx.rows());
      const Eigen::Index cols = std::min(total.dfdxx.cols(), term.dfdxx.cols());
      total.dfdxx.topLeftCorner(rows, cols).noalias() += term.dfdxx.topLeftCorner(rows, cols);
    }
  }  // namespace

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

    ocs2::ScalarFunctionQuadraticApproximation cost = ocs2::ScalarFunctionQuadraticApproximation::Zero(state_variable_dim_);
    std::for_each(firstActive, terms_.end(), [&](const std::unique_ptr<ocs2::StateCost>& costTerm) {
                                          if (costTerm->isActive(time)) {
                                            addPadded(cost, costTerm->getQuadraticApproximation(time, state, targetTrajectories, preComp));
                                          }
                                        });

    // Make sure that input derivatives are empty
    cost.dfdu = ocs2::vector_t();
    cost.dfduu = ocs2::matrix_t();
    cost.dfdux = ocs2::matrix_t();

    return cost;
  }

}
