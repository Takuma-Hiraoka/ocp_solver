#include <ocs2_core/PreComputation.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "ocp_solver/solver/dynamics_helper_functions.h"
#include "ocp_solver/solver/state_converter.h"

namespace ocp_solver {

  class OCPPreComputation : public ocs2::PreComputation {
  public:
    OCPPreComputation(ocs2::PinocchioInterface pinocchioInterface,
                      const StateConverter<ocs2::scalar_t>& stateConverter);
    ~OCPPreComputation() override = default;

    OCPPreComputation* clone() const override;

    void request(ocs2::RequestSet request, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u) override;
    ocs2::PinocchioInterface& getPinocchioInterface() { return pinocchioInterface_; }
    ocs2::PinocchioInterface& getPinocchioInterface() const { return pinocchioInterface_; }
    const ocs2::vector_t& getGeneralizedCoordinates() const { return q_; }
    const ocs2::vector_t& getGeneralizedVelocities() const { return v_; }
    const ocs2::vector_t& getGeneralizedAccelerations() const { return a_; }
    const ocs2::vector_t& getState() const { return state_; }
    const ocs2::vector_t& getInput() const { return input_; }
    const BaseAccelerationLinearApproximation& getBaseAccelerationLinearApproximation() const;

  protected:
    OCPPreComputation(const OCPPreComputation& rhs);

    void updatePinocchioModelKinematics(const ocs2::vector_t& q, const ocs2::vector_t& v, const ocs2::vector_t& a) const;

    mutable ocs2::PinocchioInterface pinocchioInterface_;
    const StateConverter<ocs2::scalar_t>* stateConverterPtr_;
    ocs2::vector_t q_;
    ocs2::vector_t v_;
    ocs2::vector_t a_;
    ocs2::vector_t state_;
    ocs2::vector_t input_;
    mutable bool baseAccelerationLinearApproximationValid_ = false;
    mutable BaseAccelerationLinearApproximation baseAccelerationLinearApproximation_;
  };

}
