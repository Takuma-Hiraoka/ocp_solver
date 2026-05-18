#pragma once

#include <pinocchio/fwd.hpp>

#include <array>
#include <cppad/cg.hpp>
#include <iostream>
#include <memory>

#include <pinocchio/algorithm/center-of-mass.hpp>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include "ocp_solver/solver/state_converter.h"

namespace ocp_solver {

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, 6, 1> computeBaseAcceleration(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                        const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                        StateConverter<SCALAR_T>& stateConverter);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeGeneralizedAccelerations(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                                 const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                                 const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                                 StateConverter<SCALAR_T>& stateConverter);

  template <typename SCALAR_T>
    Eigen::Matrix<SCALAR_T, 6, 1> computeBaseAcceleration(const Eigen::Matrix<SCALAR_T, -1, -1>& M,
                                                          const Eigen::Matrix<SCALAR_T, -1, 1>& nle,
                                                          const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                          const Eigen::Matrix<SCALAR_T, -1, 1>& externalForcesInJointSpace);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeStateDerivative(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                        const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                        StateConverter<SCALAR_T>& stateConverter);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorques(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                     const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                     ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                     StateConverter<SCALAR_T>& stateConverter);

  template <typename SCALAR_T>
    Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorques(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                           const Eigen::Matrix<SCALAR_T, -1, 1>& qd,
                                           const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                           const std::vector<std::pair<Eigen::Matrix<SCALAR_T, 6, 1>, pinocchio::FrameIndex>>& wrenches,
                                           ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

  template <typename SCALAR_T>
    Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorquesRNEA(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                               const Eigen::Matrix<SCALAR_T, -1, 1>& qd,
                                               const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                               const std::vector<std::pair<Eigen::Matrix<SCALAR_T, 6, 1>, pinocchio::FrameIndex>>& wrenches,
                                               ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface);

}
