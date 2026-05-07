#pragma once

#include <ocs2_core/Types.h>
#include <ocs2_core/integration/SensitivityIntegrator.h>
#include <ocs2_core/model_data/Metrics.h>

#include <ocs2_oc/oc_problem/OptimalControlProblem.h>

namespace ocp_solver {
  ocs2::Metrics computeIntermediateMetrics(ocs2::OptimalControlProblem& optimalControlProblem, ocs2::DynamicsDiscretizer& discretizer, ocs2::scalar_t t, ocs2::scalar_t dt,
                                           const ocs2::vector_t& x, const ocs2::vector_t& x_next, const ocs2::vector_t& u);
  ocs2::Metrics computeEventMetrics(ocs2::OptimalControlProblem& optimalControlProblem, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& x_next);
  ocs2::Metrics computeTerminalMetrics(ocs2::OptimalControlProblem& optimalControlProblem, ocs2::scalar_t t, const ocs2::vector_t& x);

}
