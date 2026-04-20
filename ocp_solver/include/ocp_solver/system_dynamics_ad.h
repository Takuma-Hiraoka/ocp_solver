#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include "ocp_solver/state_converter.h"

namespace ocp_solver {
  class SystemDynamicsAD final : public ocs2::SystemDynamicsBaseAD {
  public:
    SystemDynamicsAD(const ocs2::PinocchioInterface& pinocchioInterface,
                     StateConverter<ocs2::ad_scalar_t>& stateConverter,
                     const std::string& modelName,
                     const std::string& modelFolder,
                     const bool& recompileLibraries=false,
                     const bool& verbose=true);

    ~SystemDynamicsAD() override = default;

    SystemDynamicsAD(const SystemDynamicsAD& rhs) = default;

    SystemDynamicsAD* clone() const override { return new SystemDynamicsAD(*this); }

    ocs2::ad_vector_t systemFlowMap(ocs2::ad_scalar_t time,
                                    const ocs2::ad_vector_t& state,
                                    const ocs2::ad_vector_t& input,
                                    const ocs2::ad_vector_t& parameters) const override;

  private:
    ocs2::PinocchioInterfaceCppAd pinInterfaceCppAd;
    StateConverter<ocs2::ad_scalar_t>& stateConverter_;
  };
}
