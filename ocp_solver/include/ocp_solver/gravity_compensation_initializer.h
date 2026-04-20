#pragma once

#include <ocs2_core/initialization/Initializer.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "ocp_solver/state_converter.h"
#include "ocp_solver/switched_model_reference_manager.h"

namespace ocp_solver {

  class GravityCompensationInitializer final : public ocs2::Initializer {
  public:
    /*
     * Constructor
     * @param [in] pinocchioInterface : The pinocchio model interface
     * @param [in] referenceManager : Switched system reference manager.
     * @param [in] extendNormalizedMomentum: If true, it extrapolates the normalized momenta; otherwise sets them to zero.
     */
    GravityCompensationInitializer(const ocs2::PinocchioInterface& pinocchioInterface,
                                   const SwitchedModelReferenceManager& referenceManager,
                                   const StateConverter<ocs2::scalar_t>& stateConverter);

    ~GravityCompensationInitializer() override = default;
    GravityCompensationInitializer* clone() const override;

    void compute(ocs2::scalar_t time, const ocs2::vector_t& state, ocs2::scalar_t nextTime, ocs2::vector_t& input, ocs2::vector_t& nextState) override;

  private:
    GravityCompensationInitializer(const GravityCompensationInitializer& rhs);

    const StateConverter<ocs2::scalar_t>* stateConverterPtr_;
    const ocs2::PinocchioInterface& pinocchioInterface_;
    const SwitchedModelReferenceManager* referenceManagerPtr_;
  };

}
