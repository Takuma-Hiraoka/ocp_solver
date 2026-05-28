#include "ocp_constraint/joint_torque_cost.h"
#include <ocp_solver/solver/ocp_pre_computation.h>
#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

namespace ocp_constraint {
  namespace {
    ocs2::vector_t getJointTorques(const ocs2::vector_t& state,
                                   const ocs2::vector_t& input,
                                   ocs2::PinocchioInterface& pinocchioInterface,
                                   ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter) {
      return ocp_solver::computeJointTorques<ocs2::scalar_t>(state, input, pinocchioInterface, stateConverter);
    }

    ocs2::matrix_t externalForceDerivativeWrtConfiguration(
        const ocs2::vector_t& q,
        const ocs2::vector_t& input,
        ocs2::PinocchioInterface& pinocchioInterface,
        const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter) {
      const pinocchio::Model& model = pinocchioInterface.getModel();
      pinocchio::Data& data = pinocchioInterface.getData();
      const size_t tangentDim = stateConverter.getTangentDim();

      ocs2::matrix_t derivative = ocs2::matrix_t::Zero(tangentDim, tangentDim);
      return derivative;
    }

    ocs2::matrix_t externalForceDerivativeWrtInput(
        const ocs2::vector_t& q,
        ocs2::PinocchioInterface& pinocchioInterface,
        const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter) {
      const pinocchio::Model& model = pinocchioInterface.getModel();
      pinocchio::Data& data = pinocchioInterface.getData();
      const size_t tangentDim = stateConverter.getTangentDim();

      ocs2::matrix_t derivative = ocs2::matrix_t::Zero(tangentDim, stateConverter.getInputDim());
      for (size_t i = 0; i < stateConverter.contactCandidates.size(); i++) {
        ocs2::matrix_t jacobian = ocs2::matrix_t::Zero(6, tangentDim);
        pinocchio::computeJointJacobians(model, data, q);
        ocp_solver::getContactCandidateJacobian(pinocchioInterface, stateConverter.getContactCandidate(i),
                                                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, jacobian);
        derivative.middleCols<6>(6 * i).noalias() = jacobian.transpose();
      }
      return derivative;
    }

    std::pair<ocs2::matrix_t, ocs2::matrix_t> torqueDerivatives(
        const ocs2::vector_t& state,
        const ocs2::vector_t& input,
        const ocp_solver::OCPPreComputation& preComp,
        ocs2::PinocchioInterface& pinocchioInterface,
        const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter) {
      const pinocchio::Model& model = pinocchioInterface.getModel();
      pinocchio::Data& data = pinocchioInterface.getData();
      const size_t tangentDim = stateConverter.getTangentDim();
      const size_t baseVDim = stateConverter.getBaseVDim();
      const size_t jointDim = stateConverter.getJointDim();
      const size_t inputDim = stateConverter.getInputDim();

      const ocs2::vector_t q = stateConverter.getGeneralizedCoordinates(state);
      const ocs2::vector_t v = stateConverter.getGeneralizedVelocities(state, input);
      const ocs2::vector_t qdd = preComp.getGeneralizedAccelerations();

      ocs2::matrix_t dqdd_dx = ocs2::matrix_t::Zero(tangentDim, stateConverter.getStateVariableDim());
      ocs2::matrix_t dqdd_du = ocs2::matrix_t::Zero(tangentDim, inputDim);
      if (baseVDim > 0) {
        const ocp_solver::BaseAccelerationLinearApproximation& baseAccelerationApprox = preComp.getBaseAccelerationLinearApproximation();
        dqdd_dx.block(0, 0, baseVDim, tangentDim) = baseAccelerationApprox.dfdq;
        dqdd_dx.block(0, tangentDim, baseVDim, tangentDim) = baseAccelerationApprox.dfdv;
        dqdd_du.topRows(baseVDim) = baseAccelerationApprox.dfdu;
      }
      dqdd_du.block(baseVDim, stateConverter.getJointAccelerationsStartindex(), jointDim, jointDim).setIdentity();

      const ocs2::matrix_t externalDerivativeQ =
        externalForceDerivativeWrtConfiguration(q, input, pinocchioInterface, stateConverter);
      const ocs2::matrix_t externalDerivativeU =
        externalForceDerivativeWrtInput(q, pinocchioInterface, stateConverter);

      pinocchio::computeRNEADerivatives(model, data, q, v, qdd);
      ocs2::matrix_t dtaudx = ocs2::matrix_t::Zero(jointDim, stateConverter.getStateVariableDim());
      ocs2::matrix_t dtaudu = ocs2::matrix_t::Zero(jointDim, inputDim);
      const ocs2::matrix_t dtauda = data.M.bottomRows(jointDim);

      dtaudx.leftCols(tangentDim).noalias() =
        data.dtau_dq.bottomRows(jointDim) + dtauda * dqdd_dx.leftCols(tangentDim) -
        externalDerivativeQ.bottomRows(jointDim);
      dtaudx.block(0, tangentDim, jointDim, tangentDim).noalias() =
        data.dtau_dv.bottomRows(jointDim) + dtauda * dqdd_dx.block(0, tangentDim, tangentDim, tangentDim);
      dtaudu.noalias() = dtauda * dqdd_du - externalDerivativeU.bottomRows(jointDim);

      return {dtaudx, dtaudu};
    }
  }  // namespace

  JointTorqueCost::JointTorqueCost(const ocs2::matrix_t& weights, const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter)
    : weights_(weights),
      stateConverter_(stateConverter.clone()){}

  JointTorqueCost::JointTorqueCost(const JointTorqueCost& rhs)
    : StateInputCost(rhs),
      weights_(rhs.weights_),
      stateConverter_(rhs.stateConverter_->clone()) {}

  ocs2::scalar_t JointTorqueCost::getValue(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::TargetTrajectories& targetTrajectories,
                                           const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const ocs2::vector_t torques = getJointTorques(state, input, pinocchioInterface, *stateConverter_);

    return 0.5 * torques.dot(weights_ * torques);
  }

  ocs2::ScalarFunctionQuadraticApproximation JointTorqueCost::getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input,
                                                                                        const ocs2::TargetTrajectories& targetTrajectories,
                                                                                        const ocs2::PreComputation& preComp) const {

    ocs2::ScalarFunctionQuadraticApproximation torque;

    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const ocs2::vector_t torques = getJointTorques(state, input, pinocchioInterface, *stateConverter_);
    const std::pair<ocs2::matrix_t, ocs2::matrix_t> torqueDerivativePair = torqueDerivatives(state, input, ocpPreComp, pinocchioInterface, *stateConverter_);
    const ocs2::matrix_t& dtaudx = torqueDerivativePair.first;
    const ocs2::matrix_t& dtaudu = torqueDerivativePair.second;

    torque.f = 0.5 * torques.dot(weights_ * torques);
    torque.dfdx = dtaudx.transpose() * weights_ * torques;
    torque.dfdu = dtaudu.transpose() * weights_ * torques;
    torque.dfdxx = dtaudx.transpose() * weights_ * dtaudx;
    torque.dfduu = dtaudu.transpose() * weights_ * dtaudu;
    torque.dfdux = dtaudu.transpose() * weights_ * dtaudx;

    return torque;
  }

}
