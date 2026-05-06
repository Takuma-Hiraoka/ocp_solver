#include "ocp_solver/ocp_pinocchio_mapping.h"

namespace ocp_solver {

  template <typename SCALAR>
  OCPPinocchioMappingTpl<SCALAR>::OCPPinocchioMappingTpl(const StateConverter<SCALAR>& stateConverter)
    : stateConverterPtr_(&stateConverter) {}

  template <typename SCALAR>
  OCPPinocchioMappingTpl<SCALAR>* OCPPinocchioMappingTpl<SCALAR>::clone() const {
    return new OCPPinocchioMappingTpl<SCALAR>(*this);
  }

  template <typename SCALAR>
  auto OCPPinocchioMappingTpl<SCALAR>::getPinocchioJointPosition(const vector_t& state) const -> vector_t {
    return stateConverterPtr_->getJointAngles(state);
  }

  template <typename SCALAR>
  auto OCPPinocchioMappingTpl<SCALAR>::getPinocchioJointVelocity(const vector_t& state, const vector_t& input) const -> vector_t {
    return stateConverterPtr_->getJointVelocities(state, input);
  }

  template <typename SCALAR>
  auto OCPPinocchioMappingTpl<SCALAR>::getOcs2Jacobian(const vector_t& state, const matrix_t& Jq, const matrix_t& Jv) const -> std::pair<matrix_t, matrix_t> {
    matrix_t dfdx(Jq.rows(), stateConverterPtr_->getStateVariableDim());
    matrix_t dfdu(0, stateConverterPtr_->getInputDim());
    dfdx.leftCols(stateConverterPtr_->getTangentDim()) = Jq;
    dfdx.rightCols(stateConverterPtr_->getTangentDim()) = Jv;
    return {dfdx, dfdu};
  }

  template class ocp_solver::OCPPinocchioMappingTpl<ocs2::scalar_t>;
  template class ocp_solver::OCPPinocchioMappingTpl<ocs2::ad_scalar_t>;

}
