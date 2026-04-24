#pragma once

#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>
#include "ocp_solver/state_converter.h"

namespace ocp_solver {

  template <typename SCALAR>
    class OCPPinocchioMappingTpl;

  using OCPPinocchioMapping = OCPPinocchioMappingTpl<ocs2::scalar_t>;

  template <typename SCALAR>
    class OCPPinocchioMappingTpl final : public ocs2::PinocchioStateInputMapping<SCALAR> {
  public:
    using Base = ocs2::PinocchioStateInputMapping<SCALAR>;
    using typename Base::matrix_t;
    using typename Base::vector_t;
    explicit OCPPinocchioMappingTpl(const StateConverter<SCALAR>& stateConverter);

    ~OCPPinocchioMappingTpl() override = default;
    OCPPinocchioMappingTpl<SCALAR>* clone() const override;

    vector_t getPinocchioJointPosition(const vector_t& state) const override;
    vector_t getPinocchioJointVelocity(const vector_t& state, const vector_t& input) const override;

    std::pair<matrix_t, matrix_t> getOcs2Jacobian(const vector_t& state, const matrix_t& Jq, const matrix_t& Jv) const override;

    OCPPinocchioMappingTpl(const OCPPinocchioMappingTpl& rhs) = default;
  private:

    const StateConverter<SCALAR>* stateConverterPtr_;
  };

}
