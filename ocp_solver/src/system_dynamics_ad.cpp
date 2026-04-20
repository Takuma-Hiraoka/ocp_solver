#include "ocp_solver/system_dynamics_ad.h"
#include "ocp_solver/dynamics_helper_functions_ad.h"

namespace ocp_solver {
  SystemDynamicsAD::SystemDynamicsAD(const ocs2::PinocchioInterface& pinocchioInterface,
                               StateConverter<ocs2::ad_scalar_t>& stateConverter,
                               const std::string& modelName,
                               const std::string& modelFolder,
                               const bool& recompileLibraries,
                               const bool& verbose)
    : SystemDynamicsBaseAD(), pinInterfaceCppAd(pinocchioInterface.toCppAd()), stateConverter_(stateConverter) {
    initialize(stateConverter_.getStateDim(), stateConverter_.getInputDim(), modelName, modelFolder, recompileLibraries, verbose);
  }

  ocs2::ad_vector_t SystemDynamicsAD::systemFlowMap(ocs2::ad_scalar_t time,
                                                 const ocs2::ad_vector_t& state,
                                                 const ocs2::ad_vector_t& input,
                                                 const ocs2::ad_vector_t& parameters) const {
    return computeStateDerivative<ocs2::ad_scalar_t>(state, input, pinInterfaceCppAd, stateConverter_);
  }
}
