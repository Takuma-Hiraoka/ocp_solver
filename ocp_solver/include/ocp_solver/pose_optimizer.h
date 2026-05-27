#pragma once

#include <ocs2_core/Types.h>
#include <ocs2_oc/oc_data/PerformanceIndex.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/search_strategy/FilterLinesearch.h>
#include <ocs2_sqp/SqpSettings.h>
#include <ocs2_sqp/SqpSolverStatus.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include <ocp_solver/solver/state_converter.h>

namespace ocp_solver {

struct PoseOptimizerResult {
  ocs2::vector_t state;
  ocs2::vector_t input;
  ocs2::vector_array_t stateTrajectory;
  ocs2::vector_array_t inputTrajectory;
  ocs2::PerformanceIndex performance;
  ocs2::sqp::Convergence convergence = ocs2::sqp::Convergence::FALSE;
  int iterations = 0;
  bool success = false;
};

class PoseOptimizer {
 public:
  PoseOptimizer(const ocs2::sqp::Settings& settings,
                const ocs2::OptimalControlProblem& optimalControlProblem,
                const StateConverter<ocs2::scalar_t>& stateConverter,
                ocs2::PinocchioInterface& pinocchioInterface);

  PoseOptimizerResult run(ocs2::scalar_t time, const ocs2::vector_t& initialState, const ocs2::vector_t& initialInput);

 private:
  struct QuadraticSubproblem {
    ocs2::matrix_t hessian;
    ocs2::vector_t gradient;
    ocs2::matrix_t constraints;
    ocs2::vector_t lowerBound;
    ocs2::vector_t upperBound;
    ocs2::PerformanceIndex performance;
    ocs2::scalar_t armijoDescentMetric = 0.0;
  };

  struct StepResult {
    ocs2::sqp::StepInfo info;
    ocs2::vector_t state;
    ocs2::vector_t input;
  };

  QuadraticSubproblem setupQuadraticSubproblem(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input);
  ocs2::vector_t solveQuadraticSubproblem(const QuadraticSubproblem& subproblem) const;
  StepResult takeStep(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::vector_t& delta,
                      const QuadraticSubproblem& subproblem);
  ocs2::PerformanceIndex computePerformance(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input);
  ocs2::sqp::Convergence checkConvergence(int iteration, const ocs2::PerformanceIndex& baseline,
                                          const ocs2::sqp::StepInfo& stepInfo) const;

  void zeroVelocityAndAcceleration(ocs2::vector_t& state, ocs2::vector_t& input) const;
  ocs2::VectorFunctionLinearApproximation getQuasiStaticBalanceApproximation(const ocs2::vector_t& state,
                                                                             const ocs2::vector_t& input,
                                                                             const ocs2::PreComputation& preComp) const;
  ocs2::vector_t getQuasiStaticBalance(const ocs2::vector_t& state, const ocs2::vector_t& input) const;
  ocs2::vector_t incrementState(const ocs2::vector_t& state, const ocs2::vector_t& delta, ocs2::scalar_t alpha) const;
  ocs2::vector_t incrementInput(const ocs2::vector_t& input, const ocs2::vector_t& delta, ocs2::scalar_t alpha) const;
  ocs2::matrix_t selectStateRows(const ocs2::matrix_t& dfdx) const;
  ocs2::matrix_t selectInputRows(const ocs2::matrix_t& dfdu, Eigen::Index rows) const;
  ocs2::vector_t selectStateGradient(const ocs2::vector_t& dfdx) const;
  ocs2::vector_t selectInputGradient(const ocs2::vector_t& dfdu) const;
  void addQuadraticApproximation(const ocs2::ScalarFunctionQuadraticApproximation& approximation, ocs2::matrix_t& hessian,
                                 ocs2::vector_t& gradient, ocs2::scalar_t& cost) const;
  void appendLinearConstraint(const ocs2::VectorFunctionLinearApproximation& approximation, bool equality, ocs2::matrix_t& constraints,
                              ocs2::vector_t& lowerBound, ocs2::vector_t& upperBound) const;

  const ocs2::sqp::Settings settings_;
  const ocs2::OptimalControlProblem& problem_;
  const StateConverter<ocs2::scalar_t>& stateConverter_;
  ocs2::PinocchioInterface& pinocchioInterface_;
  ocs2::FilterLinesearch filterLinesearch_;
};

}  // namespace ocp_solver
