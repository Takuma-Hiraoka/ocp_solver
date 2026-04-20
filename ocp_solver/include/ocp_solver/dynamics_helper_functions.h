#pragma once

#include <pinocchio/fwd.hpp>

#include <array>
#include <cppad/cg.hpp>
#include <iostream>
#include <memory>

#include <pinocchio/algorithm/center-of-mass.hpp>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include "ocp_solver/state_converter.h"
#include "ocp_solver/pinocchio_frame_conversions.h"

namespace ocp_solver {
  template <typename SCALAR_T>
    void updateFramePlacements(const Eigen::Matrix<SCALAR_T, -1, 1>& q, ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

  template <typename SCALAR_T>
    void updateFramePlacements(const Eigen::Matrix<SCALAR_T, -1, 1>& q, const pinocchio::ModelTpl<SCALAR_T>& model, pinocchio::DataTpl<SCALAR_T>& data);

  template <typename SCALAR_T>
    std::vector<Eigen::Matrix<SCALAR_T, 3, 1>> computeContactPositions(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                                             ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                             const StateConverter<SCALAR_T>& stateConveter);

  template <typename SCALAR_T>
    std::vector<Eigen::Matrix<SCALAR_T, 3, 1>> getContactPositions(const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                         const StateConverter<SCALAR_T>& stateConverter);

  template <typename SCALAR_T>
    std::vector<Eigen::Matrix<SCALAR_T, 3, 1>> computeFramePositions(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                                           ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                           std::vector<std::string> frameNames);

  template <typename SCALAR_T>
    std::vector<Eigen::Matrix<SCALAR_T, 3, 1>> getFramePositions(const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                       std::vector<std::string> frameNames);

  inline size_t numberOfLegsInContacts(const size_t& contactFlags) {
    size_t numStanceLegs = 0;
    // TODO
    return numStanceLegs;
  }

  inline ocs2::vector_t gravityCompensatingInput(const ocs2::PinocchioInterface& pinocchioInterface,
                                                const size_t& contactFlags,
                                                const StateConverter<ocs2::scalar_t>& stateConverter) {
    const static ocs2::scalar_t totalGravitationalForce = computeTotalMass(pinocchioInterface.getModel()) * 9.81;
    const size_t numStanceLegs = numberOfLegsInContacts(contactFlags);
    ocs2::vector_t input = ocs2::vector_t::Zero(stateConverter.getInputDim());
    // TODO
    return input;
  }

  template <typename SCALAR_T>
    inline pinocchio::FrameIndex getContactFrameIndex(const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                      const StateConverter<SCALAR_T>& stateConverter,
                                                      size_t contactIndex) {
    return pinocchioInterface.getModel().getFrameId(stateConverter.getContactCandidates()[contactIndex].name);
  }

  template <typename SCALAR_T>
    inline std::vector<pinocchio::FrameIndex> getContactFrameIndices(const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                                     const StateConverter<SCALAR_T>& stateConverter) {
    std::vector<pinocchio::FrameIndex> contactFrameIndices;
    contactFrameIndices.reserve(stateConverter.getContactNum());
    for (size_t i = 0; i < stateConverter.getContactNum(); i++) {
      contactFrameIndices[i] = getContactFrameIndex<SCALAR_T>(pinocchioInterface, stateConverter, i);
    }
    return contactFrameIndices;
  }

  static inline Eigen::Matrix<ocs2::scalar_t, 3, 1> quaternionToEulerZYX(const Eigen::Quaternion<ocs2::scalar_t>& quat) {
    ocs2::scalar_t w = quat.w();
    ocs2::scalar_t x = quat.x();
    ocs2::scalar_t y = quat.y();
    ocs2::scalar_t z = quat.z();

    // Yaw (Z axis rotation)
    ocs2::scalar_t yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
    // Pitch (Y axis rotation)
    ocs2::scalar_t pitch = std::asin(2.0 * (w * y - z * x));
    // Roll (X axis rotation)
    ocs2::scalar_t roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));

    return Eigen::Matrix<ocs2::scalar_t, 3, 1>(yaw, pitch, roll);
  }

  template <typename SCALAR_T>
    Eigen::Matrix<SCALAR_T, 6, 1> computeBaseAcceleration(const Eigen::Matrix<SCALAR_T, -1, -1>& M,
                                                const Eigen::Matrix<SCALAR_T, -1, 1>& nle,
                                                const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                const Eigen::Matrix<SCALAR_T, -1, 1>& externalForcesInJointSpace);

  template <typename SCALAR_T>
    Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorques(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                           const Eigen::Matrix<SCALAR_T, -1, 1>& qd,
                                           const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                           const std::array<Eigen::Matrix<SCALAR_T, 6, 1>, 2>& footWrenches,
                                           ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

  ///
  /// @brief WARNING!!!!!! This formualtion currently does not work! Since pinocchio is not aware of the custom 6 dof base joint the results
  /// are wrong. Computes the joint torques via custom inverse dynamics
  ///
  /// @param q Generalized coordinates
  /// @param v Generalized velocities
  /// @param a Generalized accelerations
  /// @param footWrenches [W_left, W_right]
  /// @param pinocchioInterface
  ///
  /// @return joint torques of same dimension as qdd_joints

  template <typename SCALAR_T>
    Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorquesRNEA(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                               const Eigen::Matrix<SCALAR_T, -1, 1>& qd,
                                               const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                               const std::array<Eigen::Matrix<SCALAR_T, 6, 1>, 2>& footWrenches,
                                               ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

}
