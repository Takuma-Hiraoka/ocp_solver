#include <ocs2_oc/multiple_shooting/Helpers.h>
#include <ocs2_oc/multiple_shooting/Initialization.h>
#include <ocs2_oc/multiple_shooting/MetricsComputation.h>
#include <ocs2_oc/multiple_shooting/PerformanceIndexComputation.h>
#include <ocs2_oc/multiple_shooting/Transcription.h>
#include <ocs2_oc/oc_problem/OcpSize.h>
#include <ocs2_oc/trajectory_adjustment/TrajectorySpreadingHelperFunctions.h>
#include <algorithm>
#include <atomic>
#include "ocp_solver/solver/ocp_sqp_solver.h"
#include "ocp_solver/solver/ocp_pre_computation.h"
#include "ocp_solver/solver/ocp_transcription.h"
#include "ocp_solver/solver/ocp_sensitivity_integrator.h"
#include "ocp_solver/solver/ocp_metrics_computation.h"
#include "ocp_solver/solver/system_dynamics.h"

namespace ocp_solver {
  namespace {
    void addDynamicsViolation(ocs2::vector_t& target, const ocs2::vector_t& value) {
      if (target.size() == 0) {
        target = value;
      } else if (target.size() == value.size()) {
        target += value;
      } else {
        ocs2::vector_t resized = ocs2::vector_t::Zero(value.size());
        resized.head(std::min(target.size(), value.size())) = target.head(std::min(target.size(), value.size()));
        resized += value;
        target = std::move(resized);
      }
    }

    void initializeStateInputTrajectoriesWithTailHold(
        const ocs2::vector_t& initState,
        const std::vector<ocs2::AnnotatedTime>& timeDiscretization,
        const ocs2::PrimalSolution& primalSolution,
        ocs2::Initializer& initializer,
        ocs2::vector_array_t& stateTrajectory,
        ocs2::vector_array_t& inputTrajectory) {
      const int N = static_cast<int>(timeDiscretization.size()) - 1;
      stateTrajectory.clear();
      stateTrajectory.reserve(N + 1);
      inputTrajectory.clear();
      inputTrajectory.reserve(N);

      ocs2::scalar_t interpolateStateTill = timeDiscretization.front().time;
      ocs2::scalar_t interpolateInputTill = timeDiscretization.front().time;
      const bool hasPrimalSolution = primalSolution.timeTrajectory_.size() >= 2;
      if (hasPrimalSolution) {
        interpolateStateTill = primalSolution.timeTrajectory_.back();
        interpolateInputTill = primalSolution.timeTrajectory_[primalSolution.timeTrajectory_.size() - 2];
      }

      const ocs2::scalar_t initTime = ocs2::getIntervalStart(timeDiscretization[0]);
      if (initTime < interpolateStateTill) {
        stateTrajectory.push_back(ocs2::LinearInterpolation::interpolate(
            initTime, primalSolution.timeTrajectory_, primalSolution.stateTrajectory_));
      } else {
        stateTrajectory.push_back(initState);
      }

      for (int i = 0; i < N; ++i) {
        if (timeDiscretization[i].event == ocs2::AnnotatedTime::Event::PreEvent) {
          inputTrajectory.push_back(ocs2::vector_t());
          stateTrajectory.push_back(ocs2::multiple_shooting::initializeEventNode(
              timeDiscretization[i].time, stateTrajectory.back()));
          continue;
        }

        const ocs2::scalar_t time = ocs2::getIntervalStart(timeDiscretization[i]);
        const ocs2::scalar_t nextTime = ocs2::getIntervalEnd(timeDiscretization[i + 1]);
        ocs2::vector_t input;
        ocs2::vector_t nextState;
        if (time > interpolateInputTill || nextTime > interpolateStateTill) {
          if (hasPrimalSolution && !inputTrajectory.empty()) {
            input = inputTrajectory.back();
            nextState = stateTrajectory.back();
          } else {
            std::tie(input, nextState) =
                ocs2::multiple_shooting::initializeIntermediateNode(initializer, time, nextTime, stateTrajectory.back());
          }
        } else {
          std::tie(input, nextState) =
              ocs2::multiple_shooting::initializeIntermediateNode(primalSolution, time, nextTime);
        }
        inputTrajectory.push_back(std::move(input));
        stateTrajectory.push_back(std::move(nextState));
      }
    }
  }  // namespace

  // use pinocchio::diference
  OcpSqpSolver::OcpSqpSolver(ocs2::sqp::Settings settings, const ocs2::OptimalControlProblem& optimalControlProblem, const ocs2::Initializer& initializer)
    : SqpSolver(settings, optimalControlProblem, initializer),
      sqpIterationLimit_(std::max<size_t>(1, settings.sqpIteration)) {
    discretizer_ = ocp_solver::selectDynamicsDiscretization(settings_.integratorType);
    sensitivityDiscretizer_ = ocp_solver::selectDynamicsSensitivityDiscretization(settings_.integratorType);
  }

  void OcpSqpSolver::addStateProjection(StateProjection projection) {
    stateProjections_.push_back(std::move(projection));
  }

  void OcpSqpSolver::setSqpIteration(size_t sqpIteration) {
    sqpIterationLimit_ = std::max<size_t>(1, sqpIteration);
  }


  void OcpSqpSolver::runImpl(ocs2::scalar_t initTime, const ocs2::vector_t& initState, ocs2::scalar_t finalTime) {
    if (settings_.printSolverStatus || settings_.printLinesearch) {
      std::cerr << "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++";
      std::cerr << "\n+++++++++++++ SQP solver is initialized ++++++++++++++";
      std::cerr << "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    }

    // Determine time discretization, taking into account event times.
    const ocs2::scalar_array_t& eventTimes = this->getReferenceManager().getModeSchedule().eventTimes;
    const std::vector<ocs2::AnnotatedTime> timeDiscretization = ocs2::timeDiscretizationWithEvents(initTime, finalTime, settings_.dt, eventTimes);

    // Initialize references
    for (ocs2::OptimalControlProblem& ocpDefinition : ocpDefinitions_) {
      const ocs2::TargetTrajectories& targetTrajectories = this->getReferenceManager().getTargetTrajectories();
      ocpDefinition.targetTrajectoriesPtr = &targetTrajectories;
    }

    // Trajectory spread of primalSolution_
    if (!primalSolution_.timeTrajectory_.empty()) {
      std::ignore = ocs2::trajectorySpread(primalSolution_.modeSchedule_, this->getReferenceManager().getModeSchedule(), primalSolution_);
    }

    // Initialize the state and input
    ocs2::vector_array_t x, u;
    initializeStateInputTrajectoriesWithTailHold(initState, timeDiscretization, primalSolution_, *initializerPtr_, x, u);
    for (ocs2::vector_t& state : x) {
      for (const StateProjection& projection : stateProjections_) {
        projection(state);
      }
    }

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
      const ocs2::PerformanceIndex baselinePerformance = setupQuadraticSubproblem(timeDiscretization, initState, x, u, metrics);
      linearQuadraticApproximationTimer_.endTimer();

      // Solve QP
      solveQpTimer_.startTimer();

      // use pinocchioInterface
      ocs2::PinocchioInterface& pinocchioInterface = static_cast<const ocp_solver::OCPPreComputation&>(*(ocpDefinitions_[0].preComputationPtr)).getPinocchioInterface();
      const pinocchio::Model& model = pinocchioInterface.getModel();
      ocs2::vector_t delta_x0(initState.size() - (model.nq - model.nv));
      delta_x0.head(model.nv) = pinocchio::difference(model, x[0].head(model.nq), initState.head(model.nq));
      delta_x0.segment(model.nv, model.nv) = initState.segment(model.nq, model.nv) - x[0].segment(model.nq, model.nv);
      if (initState.size() > model.nq + model.nv) {
        const Eigen::Index extraStart = model.nq + model.nv;
        delta_x0.tail(initState.size() - extraStart) = initState.tail(initState.size() - extraStart) - x[0].tail(initState.size() - extraStart);
      }

      const ocs2::SqpSolver::OcpSubproblemSolution deltaSolution = getOCPSolution(delta_x0);
      extractValueFunction(timeDiscretization, x);
      solveQpTimer_.endTimer();

      // Apply step
      linesearchTimer_.startTimer();
      const ocs2::sqp::StepInfo stepInfo = takeStep(baselinePerformance, timeDiscretization, initState, deltaSolution, x, u, metrics);
      performanceIndeces_.push_back(stepInfo.performanceAfterStep);
      linesearchTimer_.endTimer();

      // Check convergence
      convergence = checkConvergence(iter, baselinePerformance, stepInfo);
      if (convergence == ocs2::sqp::Convergence::FALSE &&
          static_cast<size_t>(iter + 1) >= sqpIterationLimit_) {
        convergence = ocs2::sqp::Convergence::ITERATIONS;
      }

      // Logging
      if (settings_.enableLogging) {
        ocs2::sqp::LogEntry& logEntry = logger_.currentEntry();
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
    std::function<void(int)> parallelTask = [&](int workerId) {
                          // Get worker specific resources
                          ocs2::OptimalControlProblem& ocpDefinition = ocpDefinitions_[workerId];
                          ocs2::PerformanceIndex workerPerformance;  // Accumulate performance in local variable

                          int i = timeIndex++;
                          while (i < N) {
                            if (time[i].event == ocs2::AnnotatedTime::Event::PreEvent) {
                              // Event node
                              ocs2::multiple_shooting::EventTranscription result = setupEventNode(ocpDefinition, time[i].time, x[i], x[i + 1]);
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
                              ocs2::multiple_shooting::Transcription result = setupIntermediateNode(ocpDefinition, sensitivityDiscretizer_, ti, dt, x[i], x[i + 1], u[i]);
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
                            ocs2::multiple_shooting::TerminalTranscription result = ocs2::multiple_shooting::setupTerminalNode(ocpDefinition, tN, x[N]);
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
    const pinocchio::Model& model = pinocchioInterface.getModel();
    ocs2::vector_t initDynamicsViolation(initState.size() - (model.nq - model.nv));
    initDynamicsViolation.head(model.nv) = pinocchio::difference(model, x[0].head(model.nq), initState.head(model.nq));
    initDynamicsViolation.segment(model.nv, model.nv) = initState.segment(model.nq, model.nv) - x[0].segment(model.nq, model.nv);
    if (initState.size() > model.nq + model.nv) {
      const Eigen::Index extraStart = model.nq + model.nv;
      initDynamicsViolation.tail(initState.size() - extraStart) = initState.tail(initState.size() - extraStart) - x[0].tail(initState.size() - extraStart);
    }

    addDynamicsViolation(metrics.front().dynamicsViolation, initDynamicsViolation);
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
    std::function<void(int)> parallelTask = [&](int workerId) {
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
    const pinocchio::Model& model = pinocchioInterface.getModel();
    ocs2::vector_t initDynamicsViolation(initState.size() - (model.nq - model.nv));
    initDynamicsViolation.head(model.nv) = pinocchio::difference(model, x[0].head(model.nq), initState.head(model.nq));
    initDynamicsViolation.segment(model.nv, model.nv) = initState.segment(model.nq, model.nv) - x[0].segment(model.nq, model.nv);
    if (initState.size() > model.nq + model.nv) {
      const Eigen::Index extraStart = model.nq + model.nv;
      initDynamicsViolation.tail(initState.size() - extraStart) = initState.tail(initState.size() - extraStart) - x[0].tail(initState.size() - extraStart);
    }
    addDynamicsViolation(metrics.front().dynamicsViolation, initDynamicsViolation);
    performance.front().dynamicsViolationSSE += initDynamicsViolation.squaredNorm();

    // Sum performance of the threads
    ocs2::PerformanceIndex totalPerformance = std::accumulate(std::next(performance.begin()), performance.end(), performance.front());
    totalPerformance.merit = totalPerformance.cost + totalPerformance.equalityLagrangian + totalPerformance.inequalityLagrangian;
    return totalPerformance;
  }

  void OcpSqpSolver::incrementStateTrajectory(const ocs2::vector_array_t& v, const ocs2::vector_array_t& dv, const ocs2::scalar_t alpha, ocs2::vector_array_t& vNew) {
    if (v.size() != vNew.size()) throw std::runtime_error("[incrementTrajectory] Resize vNew to the size of v!");
    ocs2::PinocchioInterface& pinocchioInterface = static_cast<const ocp_solver::OCPPreComputation&>(*(ocpDefinitions_[0].preComputationPtr)).getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    for (int i = 0; i < v.size(); i++) {
      vNew[i].resize(v[i].size());
      vNew[i].head(model.nq) = pinocchio::integrate(model, v[i].head(model.nq), alpha * dv[i].head(model.nv));
      vNew[i].segment(model.nq, model.nv) = v[i].segment(model.nq, model.nv) + alpha * dv[i].segment(model.nv, model.nv);
      if (v[i].size() > model.nq + model.nv) {
        const Eigen::Index extraStart = model.nq + model.nv;
        vNew[i].tail(v[i].size() - extraStart) = v[i].tail(v[i].size() - extraStart) + alpha * dv[i].tail(v[i].size() - extraStart);
      }
      for (const StateProjection& projection : stateProjections_) {
        projection(vNew[i]);
      }
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
    const ocs2::vector_array_t& dx = subproblemSolution.deltaXSol;
    const ocs2::vector_array_t& du = subproblemSolution.deltaUSol;
    const ocs2::scalar_t deltaUnorm = ocs2::multiple_shooting::trajectoryNorm(du);
    const ocs2::scalar_t deltaXnorm = ocs2::multiple_shooting::trajectoryNorm(dx);

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
