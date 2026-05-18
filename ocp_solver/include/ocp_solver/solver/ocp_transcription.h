#pragma once

#include <ocs2_oc/multiple_shooting/Transcription.h>

namespace ocp_solver {
  ocs2::multiple_shooting::Transcription setupIntermediateNode(ocs2::OptimalControlProblem& optimalControlProblem, ocs2::DynamicsSensitivityDiscretizer& sensitivityDiscretizer,
                                    ocs2::scalar_t t, ocs2::scalar_t dt, const ocs2::vector_t& x, const ocs2::vector_t& x_next, const ocs2::vector_t& u);

  ocs2::multiple_shooting::EventTranscription setupEventNode(ocs2::OptimalControlProblem& optimalControlProblem, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& x_next);
}
