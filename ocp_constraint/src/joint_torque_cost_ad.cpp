#include <pinocchio/fwd.hpp>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "ocp_constraint/joint_torque_cost_ad.h"
#include <ocp_solver/solver/dynamics_helper_functions.h>

#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocp_constraint {
  JointTorqueCostCppAd::JointTorqueCostCppAd(const ocs2::vector_t& weights,
                                             const ocs2::PinocchioInterface& pinocchioInterface,
                                             const ocp_solver::StateConverter<ocs2::ad_scalar_t>& stateConverter,
                                             std::string costName,
                                             std::string modelFolder,
                                             bool recompileLibrariesCppAd)
    : StateInputCostGaussNewtonAd(),
      sqrtWeights_(weights.cwiseSqrt()),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      stateConverterPtr_(stateConverter.clone()) {
    assert(weights.size() == stateConverter.getJointDim());
    initialize(stateConverter.getStateVariableDim(), stateConverter.getInputDim(), stateConverter.getJointDim(), costName,
               modelFolder, recompileLibrariesCppAd);
  }

  JointTorqueCostCppAd::JointTorqueCostCppAd(const JointTorqueCostCppAd& other)
    : StateInputCostGaussNewtonAd(other),
      sqrtWeights_(other.sqrtWeights_),
      pinocchioInterfaceCppAd_(other.pinocchioInterfaceCppAd_),
      stateConverterPtr_(other.stateConverterPtr_->clone()) {}

  ocs2::ad_vector_t JointTorqueCostCppAd::costVectorFunction(ocs2::ad_scalar_t time,
                                                             const ocs2::ad_vector_t& state,
                                                             const ocs2::ad_vector_t& input,
                                                             const ocs2::ad_vector_t& parameters) const {
    return ocp_solver::computeJointTorques<ocs2::ad_scalar_t>(state, input, pinocchioInterfaceCppAd_, *stateConverterPtr_).cwiseProduct(parameters);
  }
}
