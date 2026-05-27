#include "ocp_solver/pose_optimizer.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <OsqpEigen/OsqpEigen.h>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include "ocp_solver/solver/dynamics_helper_functions.h"
#include "ocp_solver/solver/ocp_pre_computation.h"

namespace ocp_solver {
namespace {

constexpr ocs2::scalar_t kHessianRegularization = 1e-8;

void appendRows(ocs2::matrix_t& matrix, const ocs2::matrix_t& rows) {
  const Eigen::Index oldRows = matrix.rows();
  matrix.conservativeResize(oldRows + rows.rows(), Eigen::NoChange);
  matrix.bottomRows(rows.rows()) = rows;
}

void appendVector(ocs2::vector_t& vector, const ocs2::vector_t& values) {
  const Eigen::Index oldRows = vector.rows();
  vector.conservativeResize(oldRows + values.rows());
  vector.tail(values.rows()) = values;
}

}  // namespace

PoseOptimizer::PoseOptimizer(const ocs2::sqp::Settings& settings, const ocs2::OptimalControlProblem& optimalControlProblem,
                             const StateConverter<ocs2::scalar_t>& stateConverter,
                             ocs2::PinocchioInterface& pinocchioInterface)
    : settings_(settings),
      problem_(optimalControlProblem),
      stateConverter_(stateConverter),
      pinocchioInterface_(pinocchioInterface) {
  filterLinesearch_.g_max = settings_.g_max;
  filterLinesearch_.g_min = settings_.g_min;
}

PoseOptimizerResult PoseOptimizer::run(ocs2::scalar_t time, const ocs2::vector_t& initialState, const ocs2::vector_t& initialInput) {
  if (initialState.rows() != static_cast<Eigen::Index>(stateConverter_.getStateDim())) {
    throw std::runtime_error("[PoseOptimizer] initialState has invalid size.");
  }
  if (initialInput.rows() != static_cast<Eigen::Index>(stateConverter_.getInputDim())) {
    throw std::runtime_error("[PoseOptimizer] initialInput has invalid size.");
  }
  if (problem_.targetTrajectoriesPtr == nullptr) {
    throw std::runtime_error("[PoseOptimizer] OptimalControlProblem::targetTrajectoriesPtr is null.");
  }

  ocs2::vector_t state = initialState;
  ocs2::vector_t input = initialInput;
  zeroVelocityAndAcceleration(state, input);

  PoseOptimizerResult result;
  result.convergence = ocs2::sqp::Convergence::FALSE;
  result.stateTrajectory.push_back(state);
  result.inputTrajectory.push_back(input);

  int iter = 0;
  while (result.convergence == ocs2::sqp::Convergence::FALSE) {
    if (settings_.printSolverStatus || settings_.printLinesearch) {
      std::cerr << "\nPose SQP iteration: " << iter << "\n";
    }

    QuadraticSubproblem subproblem = setupQuadraticSubproblem(time, state, input);
    const ocs2::vector_t delta = solveQuadraticSubproblem(subproblem);
    subproblem.armijoDescentMetric = subproblem.gradient.dot(delta);
    const StepResult step = takeStep(time, state, input, delta, subproblem);

    state = step.state;
    input = step.input;
    result.stateTrajectory.push_back(state);
    result.inputTrajectory.push_back(input);
    result.convergence = checkConvergence(iter, subproblem.performance, step.info);
    result.performance = step.info.performanceAfterStep;

    ++iter;
  }

  result.state = state;
  result.input = input;
  result.iterations = iter;
  result.success = result.convergence != ocs2::sqp::Convergence::ITERATIONS &&
                   ocs2::FilterLinesearch::totalConstraintViolation(result.performance) < settings_.g_max;
  return result;
}

PoseOptimizer::QuadraticSubproblem PoseOptimizer::setupQuadraticSubproblem(ocs2::scalar_t time, const ocs2::vector_t& state,
                                                                           const ocs2::vector_t& input) {
  const int nqTangent = static_cast<int>(stateConverter_.getTangentDim());
  const int nw = static_cast<int>(6 * stateConverter_.getContactNum());
  const int nz = nqTangent + nw;

  QuadraticSubproblem subproblem;
  subproblem.hessian = ocs2::matrix_t::Zero(nz, nz);
  subproblem.gradient = ocs2::vector_t::Zero(nz);
  subproblem.constraints.resize(0, nz);
  subproblem.lowerBound.resize(0);
  subproblem.upperBound.resize(0);

  ocs2::PreComputation& preComp = *problem_.preComputationPtr;
  preComp.request(ocs2::Request::Cost + ocs2::Request::Constraint + ocs2::Request::SoftConstraint + ocs2::Request::Approximation, time, state,
                  input);

  ocs2::scalar_t cost = 0.0;
  addQuadraticApproximation(problem_.stateCostPtr->getQuadraticApproximation(time, state, *problem_.targetTrajectoriesPtr, preComp),
                            subproblem.hessian, subproblem.gradient, cost);
  addQuadraticApproximation(problem_.costPtr->getQuadraticApproximation(time, state, input, *problem_.targetTrajectoriesPtr, preComp),
                            subproblem.hessian, subproblem.gradient, cost);
  addQuadraticApproximation(problem_.stateSoftConstraintPtr->getQuadraticApproximation(time, state, *problem_.targetTrajectoriesPtr, preComp),
                            subproblem.hessian, subproblem.gradient, cost);
  addQuadraticApproximation(problem_.softConstraintPtr->getQuadraticApproximation(time, state, input, *problem_.targetTrajectoriesPtr, preComp),
                            subproblem.hessian, subproblem.gradient, cost);

  appendLinearConstraint(problem_.stateEqualityConstraintPtr->getLinearApproximation(time, state, preComp), true, subproblem.constraints,
                         subproblem.lowerBound, subproblem.upperBound);
  appendLinearConstraint(problem_.equalityConstraintPtr->getLinearApproximation(time, state, input, preComp), true, subproblem.constraints,
                         subproblem.lowerBound, subproblem.upperBound);
  appendLinearConstraint(getQuasiStaticBalanceApproximation(state, input, preComp), true, subproblem.constraints, subproblem.lowerBound,
                         subproblem.upperBound);
  appendLinearConstraint(problem_.stateInequalityConstraintPtr->getLinearApproximation(time, state, preComp), false, subproblem.constraints,
                         subproblem.lowerBound, subproblem.upperBound);
  appendLinearConstraint(problem_.inequalityConstraintPtr->getLinearApproximation(time, state, input, preComp), false, subproblem.constraints,
                         subproblem.lowerBound, subproblem.upperBound);

  subproblem.hessian.diagonal().array() += kHessianRegularization;
  subproblem.hessian = 0.5 * (subproblem.hessian + subproblem.hessian.transpose());
  subproblem.performance = computePerformance(time, state, input);
  return subproblem;
}

ocs2::vector_t PoseOptimizer::solveQuadraticSubproblem(const QuadraticSubproblem& subproblem) const {
  Eigen::SparseMatrix<double> hessian = subproblem.hessian.sparseView();
  Eigen::SparseMatrix<double> constraints = subproblem.constraints.sparseView();
  hessian.makeCompressed();
  constraints.makeCompressed();

  OsqpEigen::Solver solver;
  solver.settings()->setVerbosity(settings_.printSolverStatus);
  solver.settings()->setWarmStart(false);
  solver.settings()->setMaxIteration(4000);
  solver.settings()->setAbsoluteTolerance(settings_.g_min);
  solver.settings()->setRelativeTolerance(settings_.g_min);
  solver.data()->setNumberOfVariables(static_cast<int>(subproblem.gradient.rows()));
  solver.data()->setNumberOfConstraints(static_cast<int>(subproblem.lowerBound.rows()));
  solver.data()->setHessianMatrix(hessian);
  ocs2::vector_t gradient = subproblem.gradient;
  ocs2::vector_t lowerBound = subproblem.lowerBound;
  ocs2::vector_t upperBound = subproblem.upperBound;
  solver.data()->setGradient(gradient);
  solver.data()->setLinearConstraintsMatrix(constraints);
  solver.data()->setLowerBound(lowerBound);
  solver.data()->setUpperBound(upperBound);

  if (!solver.initSolver()) {
    throw std::runtime_error("[PoseOptimizer] Failed to initialize OSQP.");
  }
  const OsqpEigen::ErrorExitFlag exitFlag = solver.solveProblem();
  if (exitFlag != OsqpEigen::ErrorExitFlag::NoError) {
    throw std::runtime_error("[PoseOptimizer] OSQP failed to solve the quadratic subproblem.");
  }
  return solver.getSolution();
}

PoseOptimizer::StepResult PoseOptimizer::takeStep(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input,
                                                  const ocs2::vector_t& delta, const QuadraticSubproblem& subproblem) {
  using StepType = ocs2::FilterLinesearch::StepType;

  const int nqTangent = static_cast<int>(stateConverter_.getTangentDim());
  const ocs2::scalar_t deltaXnorm = delta.head(nqTangent).norm();
  const ocs2::scalar_t deltaUnorm = delta.tail(delta.rows() - nqTangent).norm();

  ocs2::scalar_t alpha = 1.0;
  do {
    ocs2::vector_t stateNew = incrementState(state, delta, alpha);
    ocs2::vector_t inputNew = incrementInput(input, delta, alpha);
    const ocs2::PerformanceIndex performanceNew = computePerformance(time, stateNew, inputNew);

    bool stepAccepted = false;
    StepType stepType = StepType::UNKNOWN;
    std::tie(stepAccepted, stepType) = filterLinesearch_.acceptStep(subproblem.performance, performanceNew,
                                                                    alpha * subproblem.armijoDescentMetric);

    if (settings_.printLinesearch) {
      std::cerr << "Step size: " << alpha << ", Step Type: " << ocs2::toString(stepType)
                << (stepAccepted ? " (Accepted)" : " (Rejected)") << "\n";
      std::cerr << "|dq| = " << alpha * deltaXnorm << "\t|dw| = " << alpha * deltaUnorm << "\n";
      std::cerr << performanceNew << "\n";
    }

    if (stepAccepted) {
      StepResult result;
      result.state = std::move(stateNew);
      result.input = std::move(inputNew);
      result.info.stepSize = alpha;
      result.info.stepType = stepType;
      result.info.dx_norm = alpha * deltaXnorm;
      result.info.du_norm = alpha * deltaUnorm;
      result.info.performanceAfterStep = performanceNew;
      result.info.totalConstraintViolationAfterStep = ocs2::FilterLinesearch::totalConstraintViolation(performanceNew);
      return result;
    }

    alpha *= settings_.alpha_decay;
    if (alpha * deltaXnorm < settings_.deltaTol && alpha * deltaUnorm < settings_.deltaTol) {
      break;
    }
  } while (alpha >= settings_.alpha_min);

  StepResult result;
  result.state = state;
  result.input = input;
  result.info.stepSize = 0.0;
  result.info.stepType = StepType::ZERO;
  result.info.dx_norm = 0.0;
  result.info.du_norm = 0.0;
  result.info.performanceAfterStep = subproblem.performance;
  result.info.totalConstraintViolationAfterStep = ocs2::FilterLinesearch::totalConstraintViolation(subproblem.performance);
  return result;
}

ocs2::PerformanceIndex PoseOptimizer::computePerformance(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input) {
  ocs2::PreComputation& preComp = *problem_.preComputationPtr;
  preComp.request(ocs2::Request::Cost + ocs2::Request::Constraint + ocs2::Request::SoftConstraint, time, state, input);

  ocs2::PerformanceIndex performance;
  performance.cost += problem_.stateCostPtr->getValue(time, state, *problem_.targetTrajectoriesPtr, preComp);
  performance.cost += problem_.costPtr->getValue(time, state, input, *problem_.targetTrajectoriesPtr, preComp);
  performance.cost += problem_.stateSoftConstraintPtr->getValue(time, state, *problem_.targetTrajectoriesPtr, preComp);
  performance.cost += problem_.softConstraintPtr->getValue(time, state, input, *problem_.targetTrajectoriesPtr, preComp);

  performance.equalityConstraintsSSE += ocs2::getEqConstraintsSSE(problem_.stateEqualityConstraintPtr->getValue(time, state, preComp));
  performance.equalityConstraintsSSE += ocs2::getEqConstraintsSSE(problem_.equalityConstraintPtr->getValue(time, state, input, preComp));
  performance.equalityConstraintsSSE += ocs2::getEqConstraintsSSE(getQuasiStaticBalance(state, input));
  performance.inequalityConstraintsSSE += ocs2::getIneqConstraintsSSE(problem_.stateInequalityConstraintPtr->getValue(time, state, preComp));
  performance.inequalityConstraintsSSE += ocs2::getIneqConstraintsSSE(problem_.inequalityConstraintPtr->getValue(time, state, input, preComp));
  performance.inequalityLagrangian = settings_.inequalityConstraintMu * performance.inequalityConstraintsSSE;
  performance.merit = performance.cost + performance.equalityLagrangian + performance.inequalityLagrangian;
  return performance;
}

ocs2::sqp::Convergence PoseOptimizer::checkConvergence(int iteration, const ocs2::PerformanceIndex& baseline,
                                                       const ocs2::sqp::StepInfo& stepInfo) const {
  using Convergence = ocs2::sqp::Convergence;
  if ((iteration + 1) >= settings_.sqpIteration) {
    return Convergence::ITERATIONS;
  } else if (stepInfo.stepSize < settings_.alpha_min) {
    return Convergence::STEPSIZE;
  } else if (std::abs(stepInfo.performanceAfterStep.merit - baseline.merit) < settings_.costTol &&
             ocs2::FilterLinesearch::totalConstraintViolation(stepInfo.performanceAfterStep) < settings_.g_min) {
    return Convergence::METRICS;
  } else if (stepInfo.dx_norm < settings_.deltaTol && stepInfo.du_norm < settings_.deltaTol) {
    return Convergence::PRIMAL;
  }
  return Convergence::FALSE;
}

void PoseOptimizer::zeroVelocityAndAcceleration(ocs2::vector_t& state, ocs2::vector_t& input) const {
  state.tail(pinocchioInterface_.getModel().nv).setZero();
  input.tail(stateConverter_.getJointDim()).setZero();
}

ocs2::VectorFunctionLinearApproximation PoseOptimizer::getQuasiStaticBalanceApproximation(
    const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::PreComputation& preComp) const {
  ocs2::VectorFunctionLinearApproximation approximation;
  if (stateConverter_.getBaseVDim() != 6) {
    approximation.f.resize(0);
    approximation.dfdx.resize(0, stateConverter_.getStateVariableDim());
    approximation.dfdu.resize(0, stateConverter_.getInputDim());
    return approximation;
  }

  BaseAccelerationLinearApproximation baseApproximation;
  const auto* ocpPreComputation = dynamic_cast<const OCPPreComputation*>(&preComp);
  if (ocpPreComputation != nullptr) {
    baseApproximation = ocpPreComputation->getBaseAccelerationLinearApproximation();
  } else {
    ocs2::vector_t q = stateConverter_.getGeneralizedCoordinates(state);
    ocs2::vector_t v = stateConverter_.getGeneralizedVelocities(state, input);
    ocs2::vector_t generalizedAccelerations = ocs2::vector_t::Zero(stateConverter_.getTangentDim());
    generalizedAccelerations.tail(stateConverter_.getJointDim()) = input.tail(stateConverter_.getJointDim());
    baseApproximation =
        computeBaseAccelerationLinearApproximation(q, v, generalizedAccelerations, input, pinocchioInterface_, stateConverter_);
  }

  approximation.f = getQuasiStaticBalance(state, input);
  approximation.dfdx = ocs2::matrix_t::Zero(6, stateConverter_.getStateVariableDim());
  approximation.dfdx.leftCols(stateConverter_.getTangentDim()) = baseApproximation.dfdq;
  approximation.dfdx.block(0, stateConverter_.getTangentDim(), 6, stateConverter_.getTangentDim()) = baseApproximation.dfdv;
  approximation.dfdu = baseApproximation.dfdu;
  return approximation;
}

ocs2::vector_t PoseOptimizer::getQuasiStaticBalance(const ocs2::vector_t& state, const ocs2::vector_t& input) const {
  if (stateConverter_.getBaseVDim() != 6) {
    return ocs2::vector_t(0);
  }
  StateConverter<ocs2::scalar_t>& stateConverter = const_cast<StateConverter<ocs2::scalar_t>&>(stateConverter_);
  return computeBaseAcceleration<ocs2::scalar_t>(state, input, pinocchioInterface_, stateConverter);
}

ocs2::vector_t PoseOptimizer::incrementState(const ocs2::vector_t& state, const ocs2::vector_t& delta, ocs2::scalar_t alpha) const {
  ocs2::vector_t stateNew = state;
  const int nq = pinocchioInterface_.getModel().nq;
  const int nv = pinocchioInterface_.getModel().nv;
  stateNew.head(nq) = pinocchio::integrate(pinocchioInterface_.getModel(), state.head(nq), alpha * delta.head(nv));
  stateNew.tail(nv).setZero();
  return stateNew;
}

ocs2::vector_t PoseOptimizer::incrementInput(const ocs2::vector_t& input, const ocs2::vector_t& delta, ocs2::scalar_t alpha) const {
  ocs2::vector_t inputNew = input;
  const int nqTangent = static_cast<int>(stateConverter_.getTangentDim());
  const int nw = static_cast<int>(6 * stateConverter_.getContactNum());
  if (nw > 0) {
    inputNew.head(nw) += alpha * delta.segment(nqTangent, nw);
  }
  inputNew.tail(stateConverter_.getJointDim()).setZero();
  return inputNew;
}

ocs2::matrix_t PoseOptimizer::selectStateRows(const ocs2::matrix_t& dfdx) const {
  const int nqTangent = static_cast<int>(stateConverter_.getTangentDim());
  if (dfdx.rows() == 0) {
    return ocs2::matrix_t(0, nqTangent);
  }
  return dfdx.leftCols(nqTangent);
}

ocs2::matrix_t PoseOptimizer::selectInputRows(const ocs2::matrix_t& dfdu, Eigen::Index rows) const {
  const int nw = static_cast<int>(6 * stateConverter_.getContactNum());
  if (dfdu.rows() == 0) {
    return ocs2::matrix_t::Zero(rows, nw);
  }
  return dfdu.leftCols(nw);
}

ocs2::vector_t PoseOptimizer::selectStateGradient(const ocs2::vector_t& dfdx) const {
  const int nqTangent = static_cast<int>(stateConverter_.getTangentDim());
  if (dfdx.rows() == 0) {
    return ocs2::vector_t::Zero(nqTangent);
  }
  return dfdx.head(nqTangent);
}

ocs2::vector_t PoseOptimizer::selectInputGradient(const ocs2::vector_t& dfdu) const {
  const int nw = static_cast<int>(6 * stateConverter_.getContactNum());
  if (dfdu.rows() == 0) {
    return ocs2::vector_t::Zero(nw);
  }
  return dfdu.head(nw);
}

void PoseOptimizer::addQuadraticApproximation(const ocs2::ScalarFunctionQuadraticApproximation& approximation, ocs2::matrix_t& hessian,
                                              ocs2::vector_t& gradient, ocs2::scalar_t& cost) const {
  const int nqTangent = static_cast<int>(stateConverter_.getTangentDim());
  const int nw = static_cast<int>(6 * stateConverter_.getContactNum());

  gradient.head(nqTangent) += selectStateGradient(approximation.dfdx);
  if (nw > 0) {
    gradient.tail(nw) += selectInputGradient(approximation.dfdu);
  }

  if (approximation.dfdxx.size() > 0) {
    hessian.topLeftCorner(nqTangent, nqTangent) += approximation.dfdxx.topLeftCorner(nqTangent, nqTangent);
  }
  if (nw > 0 && approximation.dfduu.size() > 0) {
    hessian.bottomRightCorner(nw, nw) += approximation.dfduu.topLeftCorner(nw, nw);
  }
  if (nw > 0 && approximation.dfdux.size() > 0) {
    const ocs2::matrix_t cross = approximation.dfdux.topLeftCorner(nw, nqTangent);
    hessian.bottomLeftCorner(nw, nqTangent) += cross;
    hessian.topRightCorner(nqTangent, nw) += cross.transpose();
  }
  cost += approximation.f;
}

void PoseOptimizer::appendLinearConstraint(const ocs2::VectorFunctionLinearApproximation& approximation, bool equality,
                                           ocs2::matrix_t& constraints, ocs2::vector_t& lowerBound,
                                           ocs2::vector_t& upperBound) const {
  const int nqTangent = static_cast<int>(stateConverter_.getTangentDim());
  const int nw = static_cast<int>(6 * stateConverter_.getContactNum());
  const int nz = nqTangent + nw;
  const int nc = static_cast<int>(approximation.f.rows());
  if (nc == 0) {
    return;
  }

  ocs2::matrix_t rows = ocs2::matrix_t::Zero(nc, nz);
  rows.leftCols(nqTangent) = selectStateRows(approximation.dfdx);
  if (nw > 0) {
    rows.rightCols(nw) = selectInputRows(approximation.dfdu, nc);
  }

  ocs2::vector_t lower(nc);
  ocs2::vector_t upper(nc);
  if (equality) {
    lower = -approximation.f;
    upper = -approximation.f;
  } else {
    lower = -approximation.f;
    upper.setConstant(OsqpEigen::INFTY);
  }

  appendRows(constraints, rows);
  appendVector(lowerBound, lower);
  appendVector(upperBound, upper);
}

}  // namespace ocp_solver
