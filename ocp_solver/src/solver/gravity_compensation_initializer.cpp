#include "ocp_solver/solver/gravity_compensation_initializer.h"

#include "ocp_solver/solver/dynamics_helper_functions.h"

namespace ocp_solver {

  GravityCompensationInitializer::GravityCompensationInitializer(const ocs2::PinocchioInterface& pinocchioInterface,
                                                                 const SwitchedModelReferenceManager& referenceManager,
                                                                 const StateConverter<ocs2::scalar_t>& stateConverter)
    : pinocchioInterface_(pinocchioInterface), referenceManagerPtr_(&referenceManager), stateConverterPtr_(&stateConverter) {}

  GravityCompensationInitializer::GravityCompensationInitializer(const GravityCompensationInitializer& rhs)
    : pinocchioInterface_(rhs.pinocchioInterface_),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      stateConverterPtr_(rhs.stateConverterPtr_) {}

  GravityCompensationInitializer* GravityCompensationInitializer::clone() const {
    return new GravityCompensationInitializer(*this);
  }

  void GravityCompensationInitializer::compute(ocs2::scalar_t time, const ocs2::vector_t& state, ocs2::scalar_t nextTime, ocs2::vector_t& input, ocs2::vector_t& nextState) {
    input = ocs2::vector_t::Zero(stateConverterPtr_->getInputDim());
    for (int i=0; i<stateConverterPtr_->getContactNum(); i++) {
      if (referenceManagerPtr_->isInContact(time, stateConverterPtr_->getContactCandidateIds()[i])) input[i*6 + 2] = 100; // TODO
    }
    nextState = state;
  }

}
