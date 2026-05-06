#include "ocp_solver/dynamics_helper_functions_ad.h"
#include "ocp_solver/dynamics_helper_functions.h"

#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>

// Pinnochio
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocp_solver {

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, 6, 1> computeBaseAcceleration(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                        const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                        StateConverter<SCALAR_T>& stateConverter) {
    const auto& model = pinInterface.getModel();
    auto data = pinInterface.getData();
    const Eigen::Matrix<SCALAR_T, -1, 1> q = stateConverter.getGeneralizedCoordinates(state);
    const Eigen::Matrix<SCALAR_T, -1, 1> qd = stateConverter.getGeneralizedVelocities(state, input);
    const Eigen::Matrix<SCALAR_T, -1, 1> qdd_joints = stateConverter.getJointAccelerations(input);

    data.M.fill(SCALAR_T(0.0));
    pinocchio::crba(model, data, q);
    pinocchio::nonLinearEffects(model, data, q, qd);

    Eigen::Matrix<SCALAR_T, 6, 1> baseExternalForces = Eigen::Matrix<SCALAR_T, 6, 1>::Zero();

    for (int i=0; i<stateConverter.contactCandidateIds.size(); i++) {
      Eigen::Matrix<SCALAR_T, -1, -1> J = Eigen::Matrix<SCALAR_T, -1, -1>::Zero(6, stateConverter.getGenCoordinatesDim());
      pinocchio::computeFrameJacobian(model, data, q, stateConverter.contactCandidateIds[i], pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
      Eigen::Matrix<SCALAR_T, 6, 6> J_b = J.block(0, 0, 6, 6);
      baseExternalForces += J_b.transpose() * stateConverter.getContactWrench(input, i);
    }

    return computeBaseAcceleration<SCALAR_T>(data.M, data.nle, qdd_joints, baseExternalForces);
  }
  template Eigen::Matrix<ocs2::ad_scalar_t, 6, 1> computeBaseAcceleration(const ocs2::ad_vector_t& state,
                                                                        const ocs2::ad_vector_t& input,
                                                                        const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinInterface,
                                                                        StateConverter<ocs2::ad_scalar_t>& stateConverter);
  template Eigen::Matrix<ocs2::scalar_t, 6, 1> computeBaseAcceleration(const ocs2::vector_t& state,
                                                                        const ocs2::vector_t& input,
                                                                        const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
                                                                        StateConverter<ocs2::scalar_t>& stateConverter);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeGeneralizedAccelerations(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                                 const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                                 const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                                 StateConverter<SCALAR_T>& stateConverter) {
    // Generalized Accelerations = [ddq_base, ddq_joints]
    Eigen::Matrix<SCALAR_T, -1, 1> generalizedAccelerations = Eigen::Matrix<SCALAR_T, -1, 1>::Zero(stateConverter.getTangentDim());
    if (stateConverter.getBaseVDim() == 6) generalizedAccelerations.head(6) = computeBaseAcceleration<SCALAR_T>(state, input, pinInterface, stateConverter);
    generalizedAccelerations.tail(stateConverter.getJointDim()) = stateConverter.getJointAccelerations(input);
    return generalizedAccelerations;
  }
  template ocs2::ad_vector_t computeGeneralizedAccelerations(const ocs2::ad_vector_t& state,
                                                             const ocs2::ad_vector_t& input,
                                                             const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinInterface,
                                                             StateConverter<ocs2::ad_scalar_t>& stateConverter);
  template ocs2::vector_t computeGeneralizedAccelerations(const ocs2::vector_t& state,
                                                          const ocs2::vector_t& input,
                                                          const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
                                                          StateConverter<ocs2::scalar_t>& stateConverter);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeStateDerivative(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                        const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                        StateConverter<SCALAR_T>& stateConverter) {
    // State derivative = [dq; ddq_base, ddq_joints]
    Eigen::Matrix<SCALAR_T, -1, 1> state_derivative = Eigen::Matrix<SCALAR_T, -1, 1>::Zero(stateConverter.getStateVariableDim());
    if (stateConverter.getBaseVDim() == 6) {
      state_derivative.head(3) = state.segment(stateConverter.getGenCoordinatesDim(), 3);
      // Derivatives of the euler angles ZYX
      state_derivative.segment(3, 3) = state.segment(stateConverter.getGenCoordinatesDim() + 3, 3);
    }
    state_derivative.segment(stateConverter.getBaseVDim(), stateConverter.getJointDim()) = stateConverter.getJointVelocities(state, input);
    state_derivative.tail(stateConverter.getGenCoordinatesDim()) =
      computeGeneralizedAccelerations<SCALAR_T>(state, input, pinInterface, stateConverter);
    return state_derivative;
  }
  template ocs2::ad_vector_t computeStateDerivative(const ocs2::ad_vector_t& state,
                                                    const ocs2::ad_vector_t& input,
                                                    const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinInterface,
                                                    StateConverter<ocs2::ad_scalar_t>& stateConverter);
  template ocs2::vector_t computeStateDerivative(const ocs2::vector_t& state,
                                                 const ocs2::vector_t& input,
                                                 const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
                                                 StateConverter<ocs2::scalar_t>& stateConverter);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorques(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                     const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                     ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                     StateConverter<SCALAR_T>& stateConverter) {
    const Eigen::Matrix<SCALAR_T, -1, 1> q = stateConverter.getGeneralizedCoordinates(state);
    const Eigen::Matrix<SCALAR_T, -1, 1> qd = stateConverter.getGeneralizedVelocities(state, input);
    const Eigen::Matrix<SCALAR_T, -1, 1> qdd_joints = stateConverter.getJointAccelerations(input);

    std::vector<std::pair<Eigen::Matrix<SCALAR_T, 6, 1>, pinocchio::FrameIndex>> wrenches(stateConverter.contactCandidateIds.size(), {Eigen::Matrix<SCALAR_T, 6, 1>::Zero(), 0});
    for (int i=0; i<wrenches.size(); i++) {
      wrenches[i].first = stateConverter.getContactWrench(input, i);
      wrenches[i].second  = stateConverter.contactCandidateIds[i];
    }

    return computeJointTorques<SCALAR_T>(q, qd, qdd_joints, wrenches, pinInterface);
  }
  template ocs2::ad_vector_t computeJointTorques(const ocs2::ad_vector_t& state,
                                                 const ocs2::ad_vector_t& input,
                                                 ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinInterface,
                                                 StateConverter<ocs2::ad_scalar_t>& stateConverter);
  template ocs2::vector_t computeJointTorques(const ocs2::vector_t& state,
                                              const ocs2::vector_t& input,
                                              ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
                                              StateConverter<ocs2::scalar_t>& stateConverter);

}
