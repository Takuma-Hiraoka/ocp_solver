#include "ocp_constraint/joint_torque_cost.h"
#include <ocp_solver/solver/ocp_pre_computation.h>
#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

namespace ocp_constraint {
  JointTorqueCost::JointTorqueCost(const ocs2::matrix_t& weights, const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter)
    : sqrtWeights_(std::move(weights)),
      stateConverter_(stateConverter.clone()){}
  ocs2::scalar_t JointTorqueCost::getValue(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input, const ocs2::TargetTrajectories& targetTrajectories,
                                           const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    std::vector<std::pair<Eigen::Matrix<ocs2::scalar_t, 6, 1>, pinocchio::FrameIndex>> wrenches(stateConverter_->contactCandidateIds.size(), {Eigen::Matrix<ocs2::scalar_t, 6, 1>::Zero(), 0});
    for (int i=0; i<wrenches.size(); i++) {
      wrenches[i].first = stateConverter_->getContactWrench(input, i);
      wrenches[i].second  = stateConverter_->contactCandidateIds[i];
    }


    const ocs2::vector_t q = state.head(stateConverter_->getGenCoordinatesDim());
    const ocs2::vector_t v = state.tail(stateConverter_->getTangentDim());
    const ocs2::vector_t a = input.tail(stateConverter_->getJointDim());
    ocs2::vector_t torques = ocp_solver::computeJointTorquesRNEA(q, v, a, wrenches, pinocchioInterface).tail(stateConverter_->getJointDim());

    return 0.5 * torques.dot(sqrtWeights_ * torques);
  }

  ocs2::ScalarFunctionQuadraticApproximation JointTorqueCost::getQuadraticApproximation(ocs2::scalar_t time, const ocs2::vector_t& state, const ocs2::vector_t& input,
                                                                                        const ocs2::TargetTrajectories& targetTrajectories,
                                                                                        const ocs2::PreComputation& preComp) const {

    ocs2::ScalarFunctionQuadraticApproximation torque;

    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    std::vector<std::pair<Eigen::Matrix<ocs2::scalar_t, 6, 1>, pinocchio::FrameIndex>> wrenches(stateConverter_->contactCandidateIds.size(), {Eigen::Matrix<ocs2::scalar_t, 6, 1>::Zero(), 0});
    for (int i=0; i<wrenches.size(); i++) {
      wrenches[i].first = stateConverter_->getContactWrench(input, i);
      wrenches[i].second  = stateConverter_->contactCandidateIds[i];
    }

    pinocchio::container::aligned_vector<pinocchio::Force> fextDesired(model.njoints, pinocchio::Force::Zero());

    auto setExternalForce = [&](const pinocchio::FrameIndex& frameIndex, size_t i) {
                              const auto jointIndex = model.frames[frameIndex].parentJoint;
                              const Eigen::Matrix<ocs2::scalar_t, 3, 1> translationJointFrameToContactFrame = model.frames[frameIndex].placement.translation();
                              const Eigen::Matrix<ocs2::scalar_t, 3, 3> rotationWorldFrameToJointFrame = data.oMi[jointIndex].rotation().transpose();
                              const Eigen::Matrix<ocs2::scalar_t, 3, 1> contactForce = rotationWorldFrameToJointFrame * wrenches[i].first.head(3);
                              const Eigen::Matrix<ocs2::scalar_t, 3, 1> contactTorque = rotationWorldFrameToJointFrame * wrenches[i].first.tail(3);
                              fextDesired[jointIndex].linear() = contactForce;
                              fextDesired[jointIndex].angular() = translationJointFrameToContactFrame.cross(contactForce) + contactTorque;
                            };

    for (int i=0; i<wrenches.size(); i++) setExternalForce(wrenches[i].second, i);

    ocs2::vector_t externalForcesInJointSpace = ocs2::vector_t::Zero(stateConverter_->getTangentDim());

    for (int i=0; i<wrenches.size(); i++) {
      ocs2::matrix_t J = ocs2::matrix_t::Zero(6, stateConverter_->getTangentDim());
      pinocchio::computeFrameJacobian(model, data, state.head(stateConverter_->getGenCoordinatesDim()), wrenches[i].second, rf, J);
      externalForcesInJointSpace += J.transpose() * wrenches[i].first;
    }

    // Repalce q with external forces in joint space.

    const ocs2::vector_t q = state.head(stateConverter_->getGenCoordinatesDim());

    const ocs2::vector_t v = state.tail(stateConverter_->getTangentDim());

    const ocs2::vector_t ddq_joint = input.tail(stateConverter_->getJointDim());
    ocs2::vector_t a(stateConverter_->getTangentDim());
    if (stateConverter_->getBaseVDim() > 0) a.head(6) = ocp_solver::computeBaseAcceleration(data.M, data.nle, ddq_joint, externalForcesInJointSpace);
    a.tail(stateConverter_->getJointDim()) = input.tail(stateConverter_->getJointDim());

    ocs2::vector_t torques = pinocchio::rnea(model, data, state.head(stateConverter_->getGenCoordinatesDim()), state.tail(stateConverter_->getTangentDim()), a, fextDesired).tail(stateConverter_->getJointDim());
    torque.f = 0.5 * torques.dot(sqrtWeights_ * torques);

    pinocchio::computeRNEADerivatives(model, data, state.head(stateConverter_->getGenCoordinatesDim()), state.tail(stateConverter_->getTangentDim()), a, fextDesired);

    ocs2::matrix_t dfdx = ocs2::matrix_t::Zero(stateConverter_->getJointDim(), stateConverter_->getStateVariableDim());
    dfdx = ocs2::matrix_t::Zero(stateConverter_->getJointDim(), stateConverter_->getStateVariableDim());
    dfdx.leftCols(stateConverter_->getTangentDim()) = data.dtau_dq.bottomRows(stateConverter_->getJointDim());
    dfdx.rightCols(stateConverter_->getTangentDim()) = data.dtau_dv.bottomRows(stateConverter_->getJointDim());
    torque.dfdx = dfdx.transpose() * sqrtWeights_ * torques;

    ocs2::matrix_t dfdu = ocs2::matrix_t::Zero(stateConverter_->getJointDim(), stateConverter_->getInputDim());
    for (int i=0; i<stateConverter_->contactCandidateIds.size(); i++) {
      ocs2::matrix_t J = ocs2::matrix_t::Zero(6, stateConverter_->getTangentDim());
      pinocchio::getFrameJacobian(model, data, stateConverter_->contactCandidateIds[i], rf, J);
      dfdu.middleCols(i*6, 6) = data.M.leftCols(stateConverter_->getBaseVDim()) * data.M.topLeftCorner(6,6).inverse() * J.block(0,0,6,6).transpose();
    }
    dfdu.rightCols(stateConverter_->getJointDim()) = data.M.rightCols(stateConverter_->getJointDim()) + data.M.leftCols(stateConverter_->getBaseVDim()) * data.M.topLeftCorner(6,6).inverse() * data.M.block(0, 6, 6, stateConverter_->getJointDim());
    torque.dfdu = dfdu.transpose() * sqrtWeights_ * torques;

    torque.dfdxx = dfdx.transpose() * sqrtWeights_ * dfdx;
    torque.dfduu = dfdu.transpose() * sqrtWeights_ * dfdu;
    torque.dfdux = dfdu.transpose() * sqrtWeights_ * dfdx;

    return torque;
  }

}
