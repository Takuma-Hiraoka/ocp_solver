#pragma once

#include <ocs2_sqp/SqpSolver.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

namespace ocp_solver {
  class OcpSqpSolver : public ocs2::SqpSolver {
  public:
    OcpSqpSolver(ocs2::sqp::Settings settings, const ocs2::OptimalControlProblem& optimalControlProblem, const ocs2::Initializer& initializer);
  private:
    void runImpl(ocs2::scalar_t initTime, const ocs2::vector_t& initState, ocs2::scalar_t finalTime) override;

    ocs2::PerformanceIndex setupQuadraticSubproblem(const std::vector<ocs2::AnnotatedTime>& time, const ocs2::vector_t& initState, const ocs2::vector_array_t& x, const ocs2::vector_array_t& u, std::vector<ocs2::Metrics>& metrics);

    ocs2::PerformanceIndex computePerformance(const std::vector<ocs2::AnnotatedTime>& time, const ocs2::vector_t& initState, const ocs2::vector_array_t& x, const ocs2::vector_array_t& u, std::vector<ocs2::Metrics>& metrics);

    void incrementStateTrajectory(const ocs2::vector_array_t& v, const ocs2::vector_array_t& dv, const ocs2::scalar_t alpha, ocs2::vector_array_t& vNew);

    ocs2::sqp::StepInfo takeStep(const ocs2::PerformanceIndex& baseline, const std::vector<ocs2::AnnotatedTime>& timeDiscretization, const ocs2::vector_t& initState,
                                 const ocs2::SqpSolver::OcpSubproblemSolution& subproblemSolution, ocs2::vector_array_t& x, ocs2::vector_array_t& u,
                                 std::vector<ocs2::Metrics>& metrics);
    const double deltaClip_ = 10;
  };
}
