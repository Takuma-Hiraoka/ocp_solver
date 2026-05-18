#include <pinocchio/fwd.hpp>

#include "ocp_solver/solver/dynamics_helper_functions.h"

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
  Eigen::Matrix<SCALAR_T, 6, 1> computeBaseAcceleration(const Eigen::Matrix<SCALAR_T, -1, -1>& M,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& nle,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& externalForcesInJointSpace) {
    auto M_bj = M.block(0, 6, 6, qdd_joints.size());
    Eigen::Matrix<SCALAR_T, 6, 1> intermediate = -nle.head(6) - M_bj * qdd_joints + externalForcesInJointSpace.head(6);
    if constexpr (std::is_same_v<SCALAR_T, ocs2::ad_scalar_t>) {
    // Due to the block diagonal structure of the generalized mass matrix corresponding to the base the base mass matrix can be split into a
    // linear and angular part. Which are both inverted separately. This does not only exploit part of the sparsity but also prevents a CppAD
    // branching error when multiplying a 6x6 matrix with a6 dim. vector.

      Eigen::Matrix<SCALAR_T, 3, 3> M_bb_lin = M.topLeftCorner(3, 3);
      Eigen::Matrix<SCALAR_T, 3, 3> M_bb_ang = M.block(3, 3, 3, 3);
      Eigen::Matrix<SCALAR_T, 3, 3> M_bb_lin_inv = M_bb_lin.inverse();
      Eigen::Matrix<SCALAR_T, 3, 3> M_bb_ang_inv = M_bb_ang.inverse();


      Eigen::Matrix<SCALAR_T, 6, 1> baseAccelerations;//
      baseAccelerations.head(3) = M_bb_lin_inv * intermediate.head(3);
      baseAccelerations.tail(3) = M_bb_ang_inv * intermediate.tail(3);

      return baseAccelerations;
    } else {
      Eigen::Matrix<SCALAR_T, 6, 1> baseAccelerations = M.topLeftCorner(6, 6).inverse() * intermediate;
      return baseAccelerations;
    }
  }
  template Eigen::Matrix<ocs2::scalar_t, 6, 1> computeBaseAcceleration(const Eigen::Matrix<ocs2::scalar_t, -1, -1>& M,
                                                                       const Eigen::Matrix<ocs2::scalar_t, -1, 1>& nle,
                                                                       const Eigen::Matrix<ocs2::scalar_t, -1, 1>& qdd_joints,
                                                                       const Eigen::Matrix<ocs2::scalar_t, -1, 1>& externalForcesInJointSpace);
  template Eigen::Matrix<ocs2::ad_scalar_t, 6, 1> computeBaseAcceleration(const Eigen::Matrix<ocs2::ad_scalar_t, -1, -1>& M,
                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& nle,
                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qdd_joints,
                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& externalForcesInJointSpace);

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
      Eigen::Matrix<SCALAR_T, -1, -1> J = Eigen::Matrix<SCALAR_T, -1, -1>::Zero(6, stateConverter.getTangentDim());
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
    state_derivative.tail(stateConverter.getTangentDim()) =
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

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorques(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                                     const Eigen::Matrix<SCALAR_T, -1, 1>& qd,
                                                     const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                     const std::vector<std::pair<Eigen::Matrix<SCALAR_T, 6, 1>, pinocchio::FrameIndex>>& wrenches,
                                                     ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
    const auto& model = pinocchioInterface.getModel();
    pinocchio::DataTpl<SCALAR_T>& data = pinocchioInterface.getData();

    pinocchio::crba(model, data, q);
    pinocchio::nonLinearEffects(model, data, q, qd);

    size_t n_qd = qd.size();
    Eigen::Matrix<SCALAR_T, -1, 1> externalForcesInJointSpace = Eigen::Matrix<SCALAR_T, -1, 1>::Zero(n_qd);

    for (int i=0; i<wrenches.size(); i++) {
      Eigen::Matrix<SCALAR_T, -1, -1> J = Eigen::Matrix<SCALAR_T, -1, -1>::Zero(6, n_qd);
      pinocchio::computeFrameJacobian(model, data, q, wrenches[i].second, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
      externalForcesInJointSpace += J.transpose() * wrenches[i].first;
    }

    Eigen::Matrix<SCALAR_T, -1, 1> q_dd(n_qd);
    if (n_qd > qdd_joints.size()) q_dd.head(6) = computeBaseAcceleration(data.M, data.nle, qdd_joints, externalForcesInJointSpace);
    q_dd.tail(qdd_joints.size()) = qdd_joints;

    size_t n_joints = qdd_joints.size();

    Eigen::Matrix<SCALAR_T, -1, 1> jointTorques =
      data.M.bottomRows(n_joints) * q_dd + data.nle.tail(n_joints) - externalForcesInJointSpace.tail(n_joints);

    // return jointTorques;
    return jointTorques;
  }
  template Eigen::Matrix<ocs2::scalar_t, -1, 1> computeJointTorques(const Eigen::Matrix<ocs2::scalar_t, -1, 1>& q,
                                                                    const Eigen::Matrix<ocs2::scalar_t, -1, 1>& qd,
                                                                    const Eigen::Matrix<ocs2::scalar_t, -1, 1>& qdd_joints,
                                                                    const std::vector<std::pair<Eigen::Matrix<ocs2::scalar_t, 6, 1>, pinocchio::FrameIndex>>& wrenches,
                                                                    ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface);
  template Eigen::Matrix<ocs2::ad_scalar_t, -1, 1> computeJointTorques(const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& q,
                                                                       const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qd,
                                                                       const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qdd_joints,
                                                                       const std::vector<std::pair<Eigen::Matrix<ocs2::ad_scalar_t, 6, 1>, pinocchio::FrameIndex>>& wrenches,
                                                                       ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorquesRNEA(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                                         const Eigen::Matrix<SCALAR_T, -1, 1>& qd,
                                                         const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                         const std::vector<std::pair<Eigen::Matrix<SCALAR_T, 6, 1>, pinocchio::FrameIndex>>& wrenches,
                                                         ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
    const auto& model = pinocchioInterface.getModel();
    auto& data = pinocchioInterface.getData();

    pinocchio::container::aligned_vector<pinocchio::Force> fextDesired(model.njoints, pinocchio::Force::Zero());

    pinocchio::forwardKinematics(model, data, q, qd);
    pinocchio::updateFramePlacements(model, data);

    auto setExternalForce = [&](const pinocchio::FrameIndex& frameIndex, size_t i) {
                              const auto jointIndex = model.frames[frameIndex].parentJoint;
                              const Eigen::Matrix<SCALAR_T, 3, 1> translationJointFrameToContactFrame = model.frames[frameIndex].placement.translation();
                              const Eigen::Matrix<SCALAR_T, 3, 3> rotationWorldFrameToJointFrame = data.oMi[jointIndex].rotation().transpose();
                              const Eigen::Matrix<SCALAR_T, 3, 1> contactForce = rotationWorldFrameToJointFrame * wrenches[i].first.head(3);
                              const Eigen::Matrix<SCALAR_T, 3, 1> contactTorque = rotationWorldFrameToJointFrame * wrenches[i].first.tail(3);
                              fextDesired[jointIndex].linear() = contactForce;
                              fextDesired[jointIndex].angular() = translationJointFrameToContactFrame.cross(contactForce) + contactTorque;
                            };

    for (int i=0; i<wrenches.size(); i++) setExternalForce(wrenches[i].second, i);

    pinocchio::crba(model, data, q);
    pinocchio::nonLinearEffects(model, data, q, qd);

    size_t n_qd = qd.size();
    Eigen::Matrix<SCALAR_T, -1, 1> externalForcesInJointSpace = Eigen::Matrix<SCALAR_T, -1, 1>::Zero(n_qd);

    for (int i=0; i<wrenches.size(); i++) {
      Eigen::Matrix<SCALAR_T, -1, -1> J = Eigen::Matrix<SCALAR_T, -1, -1>::Zero(6, n_qd);
      pinocchio::computeFrameJacobian(model, data, q, wrenches[i].second, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
      externalForcesInJointSpace += J.transpose() * wrenches[i].first;
    }

    // Repalce q with external forces in joint space.

    Eigen::Matrix<SCALAR_T, -1, 1> q_dd(n_qd);
    if (n_qd > qdd_joints.size()) q_dd.head(6) = computeBaseAcceleration(data.M, data.nle, qdd_joints, externalForcesInJointSpace);
    q_dd.tail(qdd_joints.size()) = qdd_joints;

    ocs2::vector_t torques = pinocchio::rnea(model, data, q, qd, q_dd, fextDesired);

    return torques.tail(qdd_joints.size());
  }
  template Eigen::Matrix<ocs2::scalar_t, -1, 1> computeJointTorquesRNEA(const Eigen::Matrix<ocs2::scalar_t, -1, 1>& q,
                                                                        const Eigen::Matrix<ocs2::scalar_t, -1, 1>& qd,
                                                                        const Eigen::Matrix<ocs2::scalar_t, -1, 1>& qdd_joints,
                                                                        const std::vector<std::pair<Eigen::Matrix<ocs2::scalar_t, 6, 1>, pinocchio::FrameIndex>>& wrenches,
                                                                        ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface);
  // template Eigen::Matrix<ocs2::ad_scalar_t, -1, 1> computeJointTorquesRNEA(const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& q,
  //                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qd,
  //                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qdd_joints,
  //                                                                          const std::vector<std::pair<Eigen::Matrix<ocs2::ad_scalar_t, 6, 1>, pinocchio::FrameIndex>>& wrenches,
  //                                                                          ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface);

}
