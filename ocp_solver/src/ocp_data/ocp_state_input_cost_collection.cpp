#include "ocp_solver/ocp_data/ocp_state_input_cost_collection.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <typeinfo>

namespace ocp_solver {
  namespace {
    void checkCostDimensions(const ocs2::ScalarFunctionQuadraticApproximation& term,
                             size_t stateVariableDim,
                             size_t inputVariableDim,
                             const std::type_info& termType) {
      const auto stateDim = static_cast<Eigen::Index>(stateVariableDim);
      const auto inputDim = static_cast<Eigen::Index>(inputVariableDim);
      if (term.dfdx.size() != stateDim || term.dfdu.size() != inputDim
          || term.dfdxx.rows() != stateDim || term.dfdxx.cols() != stateDim
          || term.dfduu.rows() != inputDim || term.dfduu.cols() != inputDim
          || term.dfdux.rows() != inputDim || term.dfdux.cols() != stateDim) {
        std::ostringstream message;
        message << "StateInputCostCollection received inconsistent quadratic approximation from "
                << termType.name()
                << ": dfdx=" << term.dfdx.size()
                << ", dfdu=" << term.dfdu.size()
                << ", dfdxx=" << term.dfdxx.rows() << "x" << term.dfdxx.cols()
                << ", dfduu=" << term.dfduu.rows() << "x" << term.dfduu.cols()
                << ", dfdux=" << term.dfdux.rows() << "x" << term.dfdux.cols()
                << ", expected state/input=" << stateVariableDim << "/" << inputVariableDim;
        throw std::runtime_error(message.str());
      }
    }

    void addPadded(ocs2::ScalarFunctionQuadraticApproximation& total,
                   const ocs2::ScalarFunctionQuadraticApproximation& term) {
      total.f += term.f;
      total.dfdx.head(std::min(total.dfdx.size(), term.dfdx.size())).noalias() +=
        term.dfdx.head(std::min(total.dfdx.size(), term.dfdx.size()));
      total.dfdu.head(std::min(total.dfdu.size(), term.dfdu.size())).noalias() +=
        term.dfdu.head(std::min(total.dfdu.size(), term.dfdu.size()));
      const Eigen::Index xxRows = std::min(total.dfdxx.rows(), term.dfdxx.rows());
      const Eigen::Index xxCols = std::min(total.dfdxx.cols(), term.dfdxx.cols());
      total.dfdxx.topLeftCorner(xxRows, xxCols).noalias() += term.dfdxx.topLeftCorner(xxRows, xxCols);
      const Eigen::Index uuRows = std::min(total.dfduu.rows(), term.dfduu.rows());
      const Eigen::Index uuCols = std::min(total.dfduu.cols(), term.dfduu.cols());
      total.dfduu.topLeftCorner(uuRows, uuCols).noalias() += term.dfduu.topLeftCorner(uuRows, uuCols);
      const Eigen::Index uxRows = std::min(total.dfdux.rows(), term.dfdux.rows());
      const Eigen::Index uxCols = std::min(total.dfdux.cols(), term.dfdux.cols());
      total.dfdux.topLeftCorner(uxRows, uxCols).noalias() += term.dfdux.topLeftCorner(uxRows, uxCols);
    }
  }  // namespace

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

    ocs2::ScalarFunctionQuadraticApproximation cost =
      ocs2::ScalarFunctionQuadraticApproximation::Zero(state_variable_dim_, input_variable_dim_);
    std::for_each(firstActive, terms_.end(), [&](const std::unique_ptr<ocs2::StateInputCost>& costTerm) {
                                          if (costTerm->isActive(time)) {
                                            const auto term =
                                              costTerm->getQuadraticApproximation(time, state, input, targetTrajectories, preComp);
                                            checkCostDimensions(term, state_variable_dim_, input_variable_dim_, typeid(*costTerm));
                                            addPadded(cost, term);
                                          }
                                        });

    return cost;
  }

}
