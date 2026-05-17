#include <ocs2_oc/multiple_shooting/Helpers.h>
#include <ocs2_oc/multiple_shooting/Initialization.h>
#include <ocs2_oc/multiple_shooting/MetricsComputation.h>
#include <ocs2_oc/multiple_shooting/PerformanceIndexComputation.h>
#include <ocs2_oc/multiple_shooting/Transcription.h>
#include <ocs2_oc/oc_problem/OcpSize.h>
#include <ocs2_oc/trajectory_adjustment/TrajectorySpreadingHelperFunctions.h>
#include "ocp_solver/ocp_sqp_solver.h"
#include "ocp_solver/ocp_pre_computation.h"
#include "ocp_solver/ocp_transcription.h"
#include "ocp_solver/ocp_sensitivity_integrator.h"
#include "ocp_solver/ocp_metrics_computation.h"
#include "ocp_solver/system_dynamics.h"

namespace ocp_solver {
  // use pinocchio::diference
  OcpSqpSolver::OcpSqpSolver(ocs2::sqp::Settings settings, const ocs2::OptimalControlProblem& optimalControlProblem, const ocs2::Initializer& initializer)
    : SqpSolver(settings, optimalControlProblem, initializer) {
    discretizer_ = ocp_solver::selectDynamicsDiscretization(settings_.integratorType);
    sensitivityDiscretizer_ = ocp_solver::selectDynamicsSensitivityDiscretization(settings_.integratorType);
  }


  void OcpSqpSolver::runImpl(ocs2::scalar_t initTime, const ocs2::vector_t& initState, ocs2::scalar_t finalTime) {
    if (settings_.printSolverStatus || settings_.printLinesearch) {
      std::cerr << "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++";
      std::cerr << "\n+++++++++++++ SQP solver is initialized ++++++++++++++";
      std::cerr << "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    }

    // Determine time discretization, taking into account event times.
    const auto& eventTimes = this->getReferenceManager().getModeSchedule().eventTimes;
    const auto timeDiscretization = ocs2::timeDiscretizationWithEvents(initTime, finalTime, settings_.dt, eventTimes);

    // Initialize references
    for (auto& ocpDefinition : ocpDefinitions_) {
      const auto& targetTrajectories = this->getReferenceManager().getTargetTrajectories();
      ocpDefinition.targetTrajectoriesPtr = &targetTrajectories;
    }

    // Trajectory spread of primalSolution_
    if (!primalSolution_.timeTrajectory_.empty()) {
      std::ignore = ocs2::trajectorySpread(primalSolution_.modeSchedule_, this->getReferenceManager().getModeSchedule(), primalSolution_);
    }

    // Initialize the state and input
    ocs2::vector_array_t x, u;
    ocs2::multiple_shooting::initializeStateInputTrajectories(initState, timeDiscretization, primalSolution_, *initializerPtr_, x, u);

    // Bookkeeping
    performanceIndeces_.clear();
    std::vector<ocs2::Metrics> metrics;

    int iter = 0;
    ocs2::sqp::Convergence convergence = ocs2::sqp::Convergence::FALSE;
    while (convergence == ocs2::sqp::Convergence::FALSE) {
      if (settings_.printSolverStatus || settings_.printLinesearch) {
        std::cerr << "\nSQP iteration: " << iter << "\n";
      }
      // Make QP approximation
      linearQuadraticApproximationTimer_.startTimer();
      const auto baselinePerformance = setupQuadraticSubproblem(timeDiscretization, initState, x, u, metrics);
      linearQuadraticApproximationTimer_.endTimer();

      // Solve QP
      solveQpTimer_.startTimer();

      // use pinocchioInterface
      ocs2::PinocchioInterface& pinocchioInterface = static_cast<const ocp_solver::OCPPreComputation&>(*(ocpDefinitions_[0].preComputationPtr)).getPinocchioInterface();
      ocs2::vector_t delta_x0(2*pinocchioInterface.getModel().nv);
      delta_x0.head(pinocchioInterface.getModel().nv) = pinocchio::difference(pinocchioInterface.getModel(), x[0].head(pinocchioInterface.getModel().nq), initState.head(pinocchioInterface.getModel().nq));
      delta_x0.tail(pinocchioInterface.getModel().nv) = initState.tail(pinocchioInterface.getModel().nv) - x[0].tail(pinocchioInterface.getModel().nv);

      const auto deltaSolution = getOCPSolution(delta_x0);
      extractValueFunction(timeDiscretization, x);
      solveQpTimer_.endTimer();

      // Apply step
      linesearchTimer_.startTimer();
      const auto stepInfo = takeStep(baselinePerformance, timeDiscretization, initState, deltaSolution, x, u, metrics);
      performanceIndeces_.push_back(stepInfo.performanceAfterStep);
      linesearchTimer_.endTimer();

      // Check convergence
      convergence = checkConvergence(iter, baselinePerformance, stepInfo);

      // Logging
      if (settings_.enableLogging) {
        auto& logEntry = logger_.currentEntry();
        logEntry.problemNumber = numProblems_;
        logEntry.time = initTime;
        logEntry.iteration = iter;
        logEntry.linearQuadraticApproximationTime = linearQuadraticApproximationTimer_.getLastIntervalInMilliseconds();
        logEntry.solveQpTime = solveQpTimer_.getLastIntervalInMilliseconds();
        logEntry.linesearchTime = linesearchTimer_.getLastIntervalInMilliseconds();
        logEntry.baselinePerformanceIndex = baselinePerformance;
        logEntry.totalConstraintViolationBaseline = ocs2::FilterLinesearch::totalConstraintViolation(baselinePerformance);
        logEntry.stepInfo = stepInfo;
        logEntry.convergence = convergence;
        logger_.advance();
      }

      // Next iteration
      ++iter;
      ++totalNumIterations_;
    }

    ++numProblems_;

    computeControllerTimer_.startTimer();
    primalSolution_ = toPrimalSolution(timeDiscretization, std::move(x), std::move(u));
    problemMetrics_ = ocs2::multiple_shooting::toProblemMetrics(timeDiscretization, std::move(metrics));
    computeControllerTimer_.endTimer();

    if (settings_.printSolverStatus || settings_.printLinesearch) {
      std::cerr << "\nConvergence : " << toString(convergence) << "\n";
      std::cerr << "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++";
      std::cerr << "\n+++++++++++++ SQP solver has terminated ++++++++++++++";
      std::cerr << "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    }
  }

  ocs2::PerformanceIndex OcpSqpSolver::setupQuadraticSubproblem(const std::vector<ocs2::AnnotatedTime>& time, const ocs2::vector_t& initState, const ocs2::vector_array_t& x, const ocs2::vector_array_t& u, std::vector<ocs2::Metrics>& metrics) {
    // Problem horizon
    const int N = static_cast<int>(time.size()) - 1;

    std::vector<ocs2::PerformanceIndex> performance(settings_.nThreads, ocs2::PerformanceIndex());
    cost_.resize(N + 1);
    dynamics_.resize(N);
    stateInputEqConstraints_.resize(N + 1);  // +1 because of HpipmInterface size check
    stateIneqConstraints_.resize(N + 1);
    stateInputIneqConstraints_.resize(N);
    constraintsProjection_.resize(N);
    projectionMultiplierCoefficients_.resize(N);
    metrics.resize(N + 1);

    std::atomic_int timeIndex{0};
    auto parallelTask = [&](int workerId) {
                          // Get worker specific resources
                          ocs2::OptimalControlProblem& ocpDefinition = ocpDefinitions_[workerId];
                          ocs2::PerformanceIndex workerPerformance;  // Accumulate performance in local variable

                          int i = timeIndex++;
                          while (i < N) {
                            if (time[i].event == ocs2::AnnotatedTime::Event::PreEvent) {
                              // Event node
                              auto result = setupEventNode(ocpDefinition, time[i].time, x[i], x[i + 1]);
                              metrics[i] = ocs2::multiple_shooting::computeMetrics(result);
                              workerPerformance += ocs2::multiple_shooting::computePerformanceIndex(result);
                              cost_[i] = std::move(result.cost);
                              dynamics_[i] = std::move(result.dynamics);
                              stateInputEqConstraints_[i].resize(0, x[i].size());
                              stateIneqConstraints_[i] = std::move(result.ineqConstraints);
                              stateInputIneqConstraints_[i].resize(0, x[i].size());
                              constraintsProjection_[i].resize(0, x[i].size());
                              projectionMultiplierCoefficients_[i] = ocs2::multiple_shooting::ProjectionMultiplierCoefficients();
                            } else {
                              // Normal, intermediate node
                              const ocs2::scalar_t ti = getIntervalStart(time[i]);
                              const ocs2::scalar_t dt = getIntervalDuration(time[i], time[i + 1]);
                              auto result = setupIntermediateNode(ocpDefinition, sensitivityDiscretizer_, ti, dt, x[i], x[i + 1], u[i]);
                              metrics[i] = ocs2::multiple_shooting::computeMetrics(result);
                              workerPerformance += ocs2::multiple_shooting::computePerformanceIndex(result, dt);
                              if (settings_.projectStateInputEqualityConstraints) {
                                ocs2::multiple_shooting::projectTranscription(result, settings_.extractProjectionMultiplier);
                              }
                              cost_[i] = std::move(result.cost);
                              dynamics_[i] = std::move(result.dynamics);
                              stateInputEqConstraints_[i] = std::move(result.stateInputEqConstraints);
                              stateIneqConstraints_[i] = std::move(result.stateIneqConstraints);
                              stateInputIneqConstraints_[i] = std::move(result.stateInputIneqConstraints);
                              constraintsProjection_[i] = std::move(result.constraintsProjection);
                              projectionMultiplierCoefficients_[i] = std::move(result.projectionMultiplierCoefficients);
                            }

                            i = timeIndex++;
                          }

                          if (i == N) {  // Only one worker will execute this
                            const ocs2::scalar_t tN = getIntervalStart(time[N]);
                            auto result = ocs2::multiple_shooting::setupTerminalNode(ocpDefinition, tN, x[N]);
                            metrics[i] = ocs2::multiple_shooting::computeMetrics(result);
                            workerPerformance += ocs2::multiple_shooting::computePerformanceIndex(result);
                            cost_[i] = std::move(result.cost);
                            stateInputEqConstraints_[i].resize(0, x[i].size());
                            stateIneqConstraints_[i] = std::move(result.ineqConstraints);
                          }

                          // Accumulate! Same worker might run multiple tasks
                          performance[workerId] += workerPerformance;
                        };
    runParallel(std::move(parallelTask));

    // Account for initial state in performance
    // use pinocchioInterface
    ocs2::PinocchioInterface& pinocchioInterface = static_cast<const ocp_solver::OCPPreComputation&>(*(ocpDefinitions_[0].preComputationPtr)).getPinocchioInterface();
    ocs2::vector_t initDynamicsViolation(2*pinocchioInterface.getModel().nv);
    initDynamicsViolation.head(pinocchioInterface.getModel().nv) = pinocchio::difference(pinocchioInterface.getModel(), x[0].head(pinocchioInterface.getModel().nq), initState.head(pinocchioInterface.getModel().nq));
    initDynamicsViolation.tail(pinocchioInterface.getModel().nv) = initState.tail(pinocchioInterface.getModel().nv) - x[0].tail(pinocchioInterface.getModel().nv);

    metrics.front().dynamicsViolation += initDynamicsViolation;
    performance.front().dynamicsViolationSSE += initDynamicsViolation.squaredNorm();

    // Sum performance of the threads
    ocs2::PerformanceIndex totalPerformance = std::accumulate(std::next(performance.begin()), performance.end(), performance.front());
    totalPerformance.merit = totalPerformance.cost + totalPerformance.equalityLagrangian + totalPerformance.inequalityLagrangian;

    return totalPerformance;
  }

  ocs2::PerformanceIndex OcpSqpSolver::computePerformance(const std::vector<ocs2::AnnotatedTime>& time, const ocs2::vector_t& initState, const ocs2::vector_array_t& x,
                                                          const ocs2::vector_array_t& u, std::vector<ocs2::Metrics>& metrics) {
    // Problem size
    const int N = static_cast<int>(time.size()) - 1;
    metrics.resize(N + 1);

    std::vector<ocs2::PerformanceIndex> performance(settings_.nThreads, ocs2::PerformanceIndex());
    std::atomic_int timeIndex{0};
    auto parallelTask = [&](int workerId) {
                          // Get worker specific resources
                          ocs2::OptimalControlProblem& ocpDefinition = ocpDefinitions_[workerId];

                          int i = timeIndex++;
                          while (i < N) {
                            if (time[i].event == ocs2::AnnotatedTime::Event::PreEvent) {
                              // Event node
                              metrics[i] = computeEventMetrics(ocpDefinition, time[i].time, x[i], x[i + 1]);
                              performance[workerId] += toPerformanceIndex(metrics[i]);
                            } else {
                              // Normal, intermediate node
                              const ocs2::scalar_t ti = getIntervalStart(time[i]);
                              const ocs2::scalar_t dt = getIntervalDuration(time[i], time[i + 1]);
                              metrics[i] = computeIntermediateMetrics(ocpDefinition, discretizer_, ti, dt, x[i], x[i + 1], u[i]);
                              performance[workerId] += toPerformanceIndex(metrics[i], dt);
                            }

                            i = timeIndex++;
                          }

                          if (i == N) {  // Only one worker will execute this
                            const ocs2::scalar_t tN = getIntervalStart(time[N]);
                            metrics[N] = computeTerminalMetrics(ocpDefinition, tN, x[N]);
                            performance[workerId] += toPerformanceIndex(metrics[N]);
                          }
                        };
    runParallel(std::move(parallelTask));

    // Account for initial state in performance
    ocs2::PinocchioInterface& pinocchioInterface = static_cast<const ocp_solver::OCPPreComputation&>(*(ocpDefinitions_[0].preComputationPtr)).getPinocchioInterface();
    ocs2::vector_t initDynamicsViolation(2*pinocchioInterface.getModel().nv);
    initDynamicsViolation.head(pinocchioInterface.getModel().nv) = pinocchio::difference(pinocchioInterface.getModel(), x[0].head(pinocchioInterface.getModel().nq), initState.head(pinocchioInterface.getModel().nq));
    initDynamicsViolation.tail(pinocchioInterface.getModel().nv) = initState.tail(pinocchioInterface.getModel().nv) - x[0].tail(pinocchioInterface.getModel().nv);
    metrics.front().dynamicsViolation += initDynamicsViolation;
    performance.front().dynamicsViolationSSE += initDynamicsViolation.squaredNorm();

    // Sum performance of the threads
    ocs2::PerformanceIndex totalPerformance = std::accumulate(std::next(performance.begin()), performance.end(), performance.front());
    totalPerformance.merit = totalPerformance.cost + totalPerformance.equalityLagrangian + totalPerformance.inequalityLagrangian;
    return totalPerformance;
  }

  void OcpSqpSolver::incrementStateTrajectory(const ocs2::vector_array_t& v, const ocs2::vector_array_t& dv, const ocs2::scalar_t alpha, ocs2::vector_array_t& vNew) {
    assert(v.size() == dv.size());
    if (v.size() != vNew.size()) throw std::runtime_error("[incrementTrajectory] Resize vNew to the size of v!");
    ocs2::PinocchioInterface& pinocchioInterface = static_cast<const ocp_solver::OCPPreComputation&>(*(ocpDefinitions_[0].preComputationPtr)).getPinocchioInterface();
    for (int i = 0; i < v.size(); i++) {
      vNew[i].resize(v[i].size());
      vNew[i].head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), v[i].head(pinocchioInterface.getModel().nq), alpha * dv[i].head(pinocchioInterface.getModel().nv));
      vNew[i].tail(pinocchioInterface.getModel().nv) = pinocchio::integrate(pinocchioInterface.getModel(), v[i].tail(pinocchioInterface.getModel().nv), alpha * dv[i].tail(pinocchioInterface.getModel().nv));
    }
  }

  ocs2::sqp::StepInfo OcpSqpSolver::takeStep(const ocs2::PerformanceIndex& baseline, const std::vector<ocs2::AnnotatedTime>& timeDiscretization, const ocs2::vector_t& initState,
                                       const ocs2::SqpSolver::OcpSubproblemSolution& subproblemSolution, ocs2::vector_array_t& x, ocs2::vector_array_t& u,
                                       std::vector<ocs2::Metrics>& metrics) {
    using StepType = ocs2::FilterLinesearch::StepType;

    /*
     * Filter linesearch based on:
     * "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming"
     * https://link.springer.com/article/10.1007/s10107-004-0559-y
     */
    if (settings_.printLinesearch) {
      std::cerr << std::setprecision(9) << std::fixed;
      std::cerr << "\n=== Linesearch ===\n";
      std::cerr << "Baseline:\n" << baseline << "\n";
    }

    // Baseline costs
    const ocs2::scalar_t baselineConstraintViolation = ocs2::FilterLinesearch::totalConstraintViolation(baseline);

    // Update norm
    const auto& dx = subproblemSolution.deltaXSol;
    const auto& du = subproblemSolution.deltaUSol;
    const auto deltaUnorm = ocs2::multiple_shooting::trajectoryNorm(du);
    const auto deltaXnorm = ocs2::multiple_shooting::trajectoryNorm(dx);

    ocs2::scalar_t alpha = 1.0;
    ocs2::vector_array_t xNew(x.size());
    ocs2::vector_array_t uNew(u.size());
    std::vector<ocs2::Metrics> metricsNew(metrics.size());
    do {
      // Compute step
      ocs2::multiple_shooting::incrementTrajectory(u, du, alpha, uNew);
      incrementStateTrajectory(x, dx, alpha, xNew);

      // Compute cost and constraints
      const ocs2::PerformanceIndex performanceNew = computePerformance(timeDiscretization, initState, xNew, uNew, metricsNew);

      // Step acceptance and record step type
      bool stepAccepted;
      StepType stepType;
      std::tie(stepAccepted, stepType) =
        filterLinesearch_.acceptStep(baseline, performanceNew, alpha * subproblemSolution.armijoDescentMetric);

      if (settings_.printLinesearch) {
        std::cerr << "Step size: " << alpha << ", Step Type: " << toString(stepType)
                  << (stepAccepted ? std::string{" (Accepted)"} : std::string{" (Rejected)"}) << "\n";
        std::cerr << "|dx| = " << alpha * deltaXnorm << "\t|du| = " << alpha * deltaUnorm << "\n";
        std::cerr << performanceNew << "\n";
      }

      if (stepAccepted) {  // Return if step accepted
        x = std::move(xNew);
        u = std::move(uNew);
        metrics = std::move(metricsNew);

        // Prepare step info
        ocs2::sqp::StepInfo stepInfo;
        stepInfo.stepSize = alpha;
        stepInfo.stepType = stepType;
        stepInfo.dx_norm = alpha * deltaXnorm;
        stepInfo.du_norm = alpha * deltaUnorm;
        stepInfo.performanceAfterStep = performanceNew;
        stepInfo.totalConstraintViolationAfterStep = ocs2::FilterLinesearch::totalConstraintViolation(performanceNew);
        return stepInfo;

      } else {  // Try smaller step
        alpha *= settings_.alpha_decay;

        // Detect too small step size during back-tracking to escape early. Prevents going all the way to alpha_min
        if (alpha * deltaXnorm < settings_.deltaTol && alpha * deltaUnorm < settings_.deltaTol) {
          if (settings_.printLinesearch) {
            std::cerr << "Exiting linesearch early due to too small primal steps |dx|: " << alpha * deltaXnorm
                      << ", and or |du|: " << alpha * deltaUnorm << " are below deltaTol: " << settings_.deltaTol << "\n";
          }
          break;
        }
      }
    } while (alpha >= settings_.alpha_min);

    // Alpha_min reached -> Don't take a step
    ocs2::sqp::StepInfo stepInfo;
    stepInfo.stepSize = 0.0;
    stepInfo.stepType = StepType::ZERO;
    stepInfo.dx_norm = 0.0;
    stepInfo.du_norm = 0.0;
    stepInfo.performanceAfterStep = baseline;
    stepInfo.totalConstraintViolationAfterStep = ocs2::FilterLinesearch::totalConstraintViolation(baseline);

    if (settings_.printLinesearch) {
      std::cerr << "[Linesearch terminated] Step size: " << stepInfo.stepSize << ", Step Type: " << toString(stepInfo.stepType) << "\n";
    }

    return stepInfo;

  }
}
