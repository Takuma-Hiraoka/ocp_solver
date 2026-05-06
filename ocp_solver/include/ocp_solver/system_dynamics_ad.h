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

    ocs2::PinocchioInterface& getPinocchioInterface() { return pinInterface_;}

    size_t getNumFlowMapParameters() const override { return numFlowMapParameters_; }
    ocs2::ad_vector_t systemFlowMap(ocs2::ad_scalar_t time,
                                    const ocs2::ad_vector_t& state,
                                    const ocs2::ad_vector_t& input,
                                    const ocs2::ad_vector_t& parameters) const override;

    size_t getNumJumpMapParameters() const override { return numJumpMapParameters_; }
    ocs2::ad_vector_t systemJumpMap(ocs2::ad_scalar_t time, const ocs2::ad_vector_t& state, const ocs2::ad_vector_t& parameters) const override;

    ocs2::VectorFunctionLinearApproximation linearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u,
                                                                const ocs2::PreComputation& preComputation) final;

    ocs2::VectorFunctionLinearApproximation jumpMapLinearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::PreComputation& preComputation) final;

  private:
    ocs2::PinocchioInterfaceCppAd pinInterfaceCppAd;
    ocs2::PinocchioInterface pinInterface_;
    StateConverter<ocs2::ad_scalar_t>& stateConverter_;
    size_t numFlowMapParameters_;
    size_t numJumpMapParameters_;
  };
}
