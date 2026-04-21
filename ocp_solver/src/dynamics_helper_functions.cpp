#include <pinocchio/fwd.hpp>

#include "ocp_solver/dynamics_helper_functions.h"

// Pinnochio
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocp_solver {

  template <typename SCALAR_T>
  void updateFramePlacements(const Eigen::Matrix<SCALAR_T, -1, 1>& q, ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
    const auto& model = pinocchioInterface.getModel();
    auto& data = pinocchioInterface.getData();
    updateFramePlacements(q, model, data);
  }
  template void updateFramePlacements(const ocs2::ad_vector_t& q, ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface);
  template void updateFramePlacements(const ocs2::vector_t& q, ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface);

  template <typename SCALAR_T>
  void updateFramePlacements(const Eigen::Matrix<SCALAR_T, -1, 1>& q, const pinocchio::ModelTpl<SCALAR_T>& model, pinocchio::DataTpl<SCALAR_T>& data) {
    pinocchio::forwardKinematics(model, data, q);
    updateFramePlacements(model, data);
  }
  template void updateFramePlacements(const ocs2::ad_vector_t& q,
                                      const pinocchio::ModelTpl<ocs2::ad_scalar_t>& model,
                                      pinocchio::DataTpl<ocs2::ad_scalar_t>& data);
  template void updateFramePlacements(const ocs2::vector_t& q, const pinocchio::ModelTpl<ocs2::scalar_t>& model, pinocchio::DataTpl<ocs2::scalar_t>& data);

  template <typename SCALAR_T>
  std::vector<Eigen::Matrix<SCALAR_T, 3, 1>> computeContactPositions(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                                                     ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                                     const StateConverter<SCALAR_T>& stateConverter) {
    updateFramePlacements<SCALAR_T>(q, pinocchioInterface);
    return getContactPositions<SCALAR_T>(pinocchioInterface, stateConverter);
  }
  template std::vector<Eigen::Matrix<ocs2::ad_scalar_t, 3, 1>> computeContactPositions(const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& q,
                                                                                       ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface,
                                                                                       const StateConverter<ocs2::ad_scalar_t>& stateConverter);
  template std::vector<Eigen::Matrix<ocs2::scalar_t, 3, 1>> computeContactPositions(const Eigen::Matrix<ocs2::scalar_t, -1, 1>& q,
                                                                                    ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface,
                                                                                    const StateConverter<ocs2::scalar_t>& stateConverter);

  template <typename SCALAR_T>
  std::vector<Eigen::Matrix<SCALAR_T, 3, 1>> getContactPositions(const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                                 const StateConverter<SCALAR_T>& stateConverter) {
    assert(stateConverter.modelSettings.contactNames.size() == stateConverter.getContactNum());
    std::vector<Eigen::Matrix<SCALAR_T, 3, 1>> contactPositions;
    contactPositions.reserve(stateConverter.getContactNum());
    const auto& data = pinocchioInterface.getData();
    std::vector<pinocchio::FrameIndex> contactFrameIndices = getContactFrameIndices(pinocchioInterface, stateConverter);

    for (size_t i = 0; i < stateConverter.getContactNum(); i++) {
      const Eigen::Matrix<SCALAR_T, 3, 1>& contactPosition = data.oMf[getContactFrameIndex(pinocchioInterface, stateConverter, i)].translation();
      contactPositions.emplace_back(contactPosition);
    }
    return contactPositions;
  }
  template std::vector<Eigen::Matrix<ocs2::ad_scalar_t, 3, 1>> getContactPositions(const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface,
                                                                                   const StateConverter<ocs2::ad_scalar_t>& stateConverter);
  template std::vector<Eigen::Matrix<ocs2::scalar_t, 3, 1>> getContactPositions(const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface,
                                                                                const StateConverter<ocs2::scalar_t>& stateConverter);

  template <typename SCALAR_T>
  std::vector<Eigen::Matrix<SCALAR_T, 3, 1>> computeFramePositions(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                                                   ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                                   std::vector<std::string> frameNames) {
    updateFramePlacements<SCALAR_T>(q, pinocchioInterface);
    return getFramePositions<SCALAR_T>(pinocchioInterface, frameNames);
  }
  template std::vector<Eigen::Matrix<ocs2::ad_scalar_t, 3, 1>> computeFramePositions(const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& q,
                                                                                     ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface,
                                                                                     std::vector<std::string> frameNames);
  template std::vector<Eigen::Matrix<ocs2::scalar_t, 3, 1>> computeFramePositions(const Eigen::Matrix<ocs2::scalar_t, -1, 1>& q,
                                                                                  ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface,
                                                                                  std::vector<std::string> frameNames);

  template <typename SCALAR_T>
  std::vector<Eigen::Matrix<SCALAR_T, 3, 1>> getFramePositions(const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                               std::vector<std::string> frameNames) {
    std::vector<Eigen::Matrix<SCALAR_T, 3, 1>> positions;
    positions.reserve(frameNames.size());
    const auto& data = pinocchioInterface.getData();
    for (size_t i = 0; i < frameNames.size(); i++) {
      const pinocchio::FrameIndex frameIndex = pinocchioInterface.getModel().getFrameId(frameNames[i]);
      const Eigen::Matrix<SCALAR_T, 3, 1>& position = data.oMf[frameIndex].translation();
      positions.emplace_back(position);
    }
    return positions;
  }
  template std::vector<Eigen::Matrix<ocs2::ad_scalar_t, 3, 1>> getFramePositions(const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface,
                                                                                 std::vector<std::string> frameNames);
  template std::vector<Eigen::Matrix<ocs2::scalar_t, 3, 1>> getFramePositions(const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface,
                                                                              std::vector<std::string> frameNames);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, 6, 1> computeBaseAcceleration(const Eigen::Matrix<SCALAR_T, -1, -1>& M,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& nle,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& externalForcesInJointSpace) {
    // Due to the block diagonal structure of the generalized mass matrix corresponding to the base the base mass matrix can be split into a
    // linear and angular part. Which are both inverted separately. This does not only exploit part of the sparsity but also prevents a CppAD
    // branching error when multiplying a 6x6 matrix with a6 dim. vector.

    Eigen::Matrix<SCALAR_T, 3, 3> M_bb_lin = M.topLeftCorner(3, 3);
    Eigen::Matrix<SCALAR_T, 3, 3> M_bb_ang = M.block(3, 3, 3, 3);
    auto M_bj = M.block(0, 6, 6, qdd_joints.size());
    Eigen::Matrix<SCALAR_T, 3, 3> M_bb_lin_inv = M_bb_lin.inverse();
    Eigen::Matrix<SCALAR_T, 3, 3> M_bb_ang_inv = M_bb_ang.inverse();

    Eigen::Matrix<SCALAR_T, 6, 1> intermediate = -nle.head(6) - M_bj * qdd_joints + externalForcesInJointSpace.head(6);

    Eigen::Matrix<SCALAR_T, 6, 1> baseAccelerations;
    baseAccelerations.head(3) = M_bb_lin_inv * intermediate.head(3);
    baseAccelerations.tail(3) = M_bb_ang_inv * intermediate.tail(3);

    return baseAccelerations;
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
  Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorques(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                                     const Eigen::Matrix<SCALAR_T, -1, 1>& qd,
                                                     const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                     const std::vector<std::pair<Eigen::Matrix<SCALAR_T, 6, 1>, std::string>>& wrenches,
                                                     ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
    const auto& model = pinocchioInterface.getModel();
    pinocchio::DataTpl<SCALAR_T>& data = pinocchioInterface.getData();

    pinocchio::crba(model, data, q);
    pinocchio::nonLinearEffects(model, data, q, qd);

    size_t n_qd = qd.size();
    Eigen::Matrix<SCALAR_T, -1, 1> externalForcesInJointSpace = Eigen::Matrix<SCALAR_T, -1, 1>::Zero(n_qd);

    for (int i=0; i<wrenches.size(); i++) {
      Eigen::Matrix<SCALAR_T, -1, -1> J = Eigen::Matrix<SCALAR_T, -1, -1>::Zero(6, n_qd);
      pinocchio::computeFrameJacobian(model, data, q, model.getFrameId(wrenches[i].second), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
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
                                                                    const std::vector<std::pair<Eigen::Matrix<ocs2::scalar_t, 6, 1>, std::string>>& wrenches,
                                                                    ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface);
  template Eigen::Matrix<ocs2::ad_scalar_t, -1, 1> computeJointTorques(const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& q,
                                                                       const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qd,
                                                                       const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qdd_joints,
                                                                       const std::vector<std::pair<Eigen::Matrix<ocs2::ad_scalar_t, 6, 1>, std::string>>& wrenches,
                                                                       ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorquesRNEA(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                                         const Eigen::Matrix<SCALAR_T, -1, 1>& qd,
                                                         const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                         const std::vector<std::pair<Eigen::Matrix<SCALAR_T, 6, 1>, std::string>>& wrenches,
                                                         ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
    const auto& model = pinocchioInterface.getModel();
    auto& data = pinocchioInterface.getData();

    pinocchio::container::aligned_vector<pinocchio::Force> fextDesired(model.njoints, pinocchio::Force::Zero());

    pinocchio::forwardKinematics(model, data, q, qd);
    pinocchio::updateFramePlacements(model, data);

    auto setExternalForce = [&](const std::string& frameName, size_t i) {
                              const auto frameIndex = model.getFrameId(frameName);
                              const auto jointIndex = model.frames[frameIndex].parent;
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
      pinocchio::computeFrameJacobian(model, data, q, model.getFrameId(wrenches[i].second), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
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
                                                                        const std::vector<std::pair<Eigen::Matrix<ocs2::scalar_t, 6, 1>, std::string>>& wrenches,
                                                                        ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface);
  // template Eigen::Matrix<ocs2::ad_scalar_t, -1, 1> computeJointTorquesRNEA(const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& q,
  //                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qd,
  //                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qdd_joints,
  //                                                                          const std::vector<std::pair<Eigen::Matrix<ocs2::ad_scalar_t, 6, 1>, std::string>>& wrenches,
  //                                                                          ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface);

}
