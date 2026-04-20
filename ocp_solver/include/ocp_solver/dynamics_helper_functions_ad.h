#pragma once

#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "ocp_solver/state_converter.h"

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
  Eigen::Matrix<SCALAR_T, -1, 1> computeStateDerivative(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                        const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                        StateConverter<SCALAR_T>& stateConverter);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorques(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                     const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                     ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                     StateConverter<SCALAR_T>& stateConverter);

}
