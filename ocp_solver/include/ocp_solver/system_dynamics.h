#pragma once

#include <ocs2_core/dynamics/SystemDynamicsBase.h>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include "ocp_solver/state_converter.h"

namespace ocp_solver {
  class SystemDynamics final : public ocs2::SystemDynamicsBase {
  public:
    SystemDynamics(const ocs2::PinocchioInterface& pinocchioInterface,
                     StateConverter<ocs2::scalar_t>& stateConverter);

    ~SystemDynamics() override = default;

    SystemDynamics(const SystemDynamics& rhs) = default;

    SystemDynamics* clone() const override { return new SystemDynamics(*this); }

    ocs2::PinocchioInterface& getPinocchioInterface() { return pinInterface_;}

    ocs2::vector_t computeFlowMap(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::PreComputation& preComputation) final;

    ocs2::vector_t computeJumpMap(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::PreComputation& preComputation) final;

    ocs2::VectorFunctionLinearApproximation linearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u,
                                                                const ocs2::PreComputation& preComputation) final;

    ocs2::VectorFunctionLinearApproximation jumpMapLinearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::PreComputation& preComputation) final;

  private:
    ocs2::PinocchioInterface pinInterface_;
    StateConverter<ocs2::scalar_t>& stateConverter_;
  };
}
