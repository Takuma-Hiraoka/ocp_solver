#pragma once

#include <ocs2_oc/oc_problem/OptimalControlProblem.h>

namespace ocp_solver {

  struct OptimalControlProblem : ocs2::OptimalControlProblem {
    OptimalControlProblem(const size_t& ss, const size_t& is);

    ~OptimalControlProblem() = default;

    OptimalControlProblem(const OptimalControlProblem& other);

    OptimalControlProblem& operator=(const OptimalControlProblem& rhs);

    void swap(OptimalControlProblem& other) noexcept;
  };

}
