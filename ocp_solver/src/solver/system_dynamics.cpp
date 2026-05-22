#include "ocp_solver/solver/system_dynamics.h"
#include "ocp_solver/solver/dynamics_helper_functions.h"
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

namespace ocp_solver {
  SystemDynamics::SystemDynamics(const ocs2::PinocchioInterface& pinocchioInterface,
                                 StateConverter<ocs2::scalar_t>& stateConverter)
    : SystemDynamicsBase(),
      pinInterface_(pinocchioInterface),
      stateConverter_(stateConverter) {}

  ocs2::vector_t SystemDynamics::computeFlowMap(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::PreComputation& preComputation) {
    return computeStateDerivative<ocs2::scalar_t>(x, u, pinInterface_, stateConverter_);
  }

  ocs2::vector_t SystemDynamics::computeJumpMap(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::PreComputation& preComputation) {
    return x;
  }

  ocs2::VectorFunctionLinearApproximation SystemDynamics::linearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u,
                                                                              const ocs2::PreComputation& preComputation) {

    const auto& model = pinInterface_.getModel();
    auto& data = pinInterface_.getData();

    const size_t tangentDim = stateConverter_.getTangentDim();
    const size_t jointDim = stateConverter_.getJointDim();
    const ocs2::vector_t q = stateConverter_.getGeneralizedCoordinates(x);
    const ocs2::vector_t v = stateConverter_.getGeneralizedVelocities(x, u);
    const ocs2::vector_t qddJoints = stateConverter_.getJointAccelerations(u);

    ocs2::VectorFunctionLinearApproximation approximation;
    approximation.dfdx.setZero(stateConverter_.getStateVariableDim(), stateConverter_.getStateVariableDim());
    approximation.dfdx.topRightCorner(tangentDim, tangentDim) = ocs2::matrix_t::Identity(tangentDim, tangentDim);
    approximation.dfdu.setZero(stateConverter_.getStateVariableDim(), u.rows());
    approximation.dfdu.bottomRightCorner(jointDim, jointDim) = ocs2::matrix_t::Identity(jointDim, jointDim);
    approximation.f = computeStateDerivative<ocs2::scalar_t>(x, u, pinInterface_, stateConverter_);

    if (stateConverter_.getBaseVDim() == 6) {
      ocs2::vector_t qdd(tangentDim);
      qdd.head(6) = approximation.f.segment(tangentDim, 6);
      qdd.tail(jointDim) = qddJoints;

      pinocchio::computeRNEADerivatives(model, data, q, v, qdd);
      const ocs2::matrix_t dtau_dq = data.dtau_dq;
      const ocs2::matrix_t dtau_dv = data.dtau_dv;
      ocs2::matrix_t M = data.M;
      M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();

      ocs2::matrix_t externalForcesDerivative = ocs2::matrix_t::Zero(tangentDim, tangentDim);
      for (size_t column = 0; column < tangentDim; ++column) {
        ocs2::vector_t tangentDirection = ocs2::vector_t::Zero(tangentDim);
        tangentDirection(column) = 1.0;
        pinocchio::computeJointJacobiansTimeVariation(model, data, q, tangentDirection);

        for (size_t i = 0; i < stateConverter_.contactCandidateIds.size(); ++i) {
          ocs2::matrix_t dJ = ocs2::matrix_t::Zero(6, tangentDim);
          pinocchio::getFrameJacobianTimeVariation(model, data, stateConverter_.contactCandidateIds[i],
                                                   pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, dJ);
          externalForcesDerivative.col(column).noalias() += dJ.transpose() * stateConverter_.getContactWrench(u, i);
        }
      }

      const ocs2::matrix_t M_bb = M.topLeftCorner(6, 6);
      const auto MbbSolver = M_bb.ldlt();
      const ocs2::matrix_t baseResidualDerivativeQ = dtau_dq.topRows(6) - externalForcesDerivative.topRows(6);

      approximation.dfdx.block(tangentDim, 0, 6, tangentDim) = -MbbSolver.solve(baseResidualDerivativeQ);
      approximation.dfdx.block(tangentDim, tangentDim, 6, tangentDim) = -MbbSolver.solve(dtau_dv.topRows(6));

      pinocchio::computeJointJacobians(model, data, q);
      for (size_t i = 0; i < stateConverter_.contactCandidateIds.size(); ++i) {
        ocs2::matrix_t J = ocs2::matrix_t::Zero(6, tangentDim);
        pinocchio::getFrameJacobian(model, data, stateConverter_.contactCandidateIds[i], pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
        approximation.dfdu.block(tangentDim, i * 6, 6, 6) = MbbSolver.solve(J.block(0, 0, 6, 6).transpose());
      }

      approximation.dfdu.block(tangentDim, stateConverter_.getJointAccelerationsStartindex(), 6, jointDim) =
        -MbbSolver.solve(M.block(0, 6, 6, jointDim));
    }

    return approximation;
  }

  ocs2::VectorFunctionLinearApproximation SystemDynamics::jumpMapLinearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x,
                                                                                     const ocs2::PreComputation& preComputation) {
    ocs2::VectorFunctionLinearApproximation approximation;
    approximation.dfdx = ocs2::matrix_t::Identity(stateConverter_.getStateVariableDim(), stateConverter_.getStateVariableDim());
    approximation.dfdu.setZero(x.rows(), 0);
    approximation.f = x;
    return approximation;
  }
}
