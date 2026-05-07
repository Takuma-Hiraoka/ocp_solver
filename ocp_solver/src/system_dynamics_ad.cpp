#include "ocp_solver/system_dynamics_ad.h"
#include "ocp_solver/dynamics_helper_functions_ad.h"

namespace ocp_solver {
  // parameterにqを入れる
  SystemDynamicsAD::SystemDynamicsAD(const ocs2::PinocchioInterface& pinocchioInterface,
                               StateConverter<ocs2::ad_scalar_t>& stateConverter,
                               const std::string& modelName,
                               const std::string& modelFolder,
                               const bool& recompileLibraries,
                               const bool& verbose)
    : SystemDynamicsBaseAD(),
      pinInterface_(pinocchioInterface),
      pinInterfaceCppAd(pinocchioInterface.toCppAd()),
      stateConverter_(stateConverter),
      numFlowMapParameters_(stateConverter_.getStateDim()),
      numJumpMapParameters_(stateConverter_.getStateDim()) {
    initialize(stateConverter_.getStateVariableDim(), stateConverter_.getInputDim(), modelName, modelFolder, recompileLibraries, verbose);
  }

  ocs2::ad_vector_t SystemDynamicsAD::systemFlowMap(ocs2::ad_scalar_t time,
                                                 const ocs2::ad_vector_t& state,
                                                 const ocs2::ad_vector_t& input,
                                                 const ocs2::ad_vector_t& parameters) const {
    ocs2::ad_vector_t q_v_state(pinInterfaceCppAd.getModel().nq + pinInterfaceCppAd.getModel().nv);
    q_v_state.head(pinInterfaceCppAd.getModel().nq) = pinocchio::integrate(pinInterfaceCppAd.getModel(), parameters.head(pinInterfaceCppAd.getModel().nq), state.head(pinInterfaceCppAd.getModel().nv));
    q_v_state.tail(pinInterfaceCppAd.getModel().nv) = parameters.tail(pinInterfaceCppAd.getModel().nv) + state.tail(pinInterfaceCppAd.getModel().nv);
    return computeStateDerivative<ocs2::ad_scalar_t>(q_v_state, input, pinInterfaceCppAd, stateConverter_);
  }

  ocs2::ad_vector_t SystemDynamicsAD::systemJumpMap(ocs2::ad_scalar_t time, const ocs2::ad_vector_t& state, const ocs2::ad_vector_t& parameters) const {
    ocs2::ad_vector_t q_v_state(pinInterfaceCppAd.getModel().nq + pinInterfaceCppAd.getModel().nv);
    q_v_state.head(pinInterfaceCppAd.getModel().nq) = pinocchio::integrate(pinInterfaceCppAd.getModel(), parameters.head(pinInterfaceCppAd.getModel().nq), state.head(pinInterfaceCppAd.getModel().nv));
    q_v_state.tail(pinInterfaceCppAd.getModel().nv) = parameters.tail(pinInterfaceCppAd.getModel().nv) + state.tail(pinInterfaceCppAd.getModel().nv);
    return q_v_state;
  }

  ocs2::vector_t SystemDynamicsAD::computeFlowMap(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::PreComputation& preComputation) {
    ocs2::vector_t dx = ocs2::vector_t::Zero(2*pinInterface_.getModel().nv);
    tapedTimeStateInput_ << t, dx, u;
    return flowMapADInterfacePtr_->getFunctionValue(tapedTimeStateInput_, x);
  }

  ocs2::vector_t SystemDynamicsAD::computeJumpMap(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::PreComputation& preComputation) {
    ocs2::vector_t dx = ocs2::vector_t::Zero(2*pinInterface_.getModel().nv);
    tapedTimeState_ << t, dx;
  return jumpMapADInterfacePtr_->getFunctionValue(tapedTimeState_, x);
  }

  ocs2::VectorFunctionLinearApproximation SystemDynamicsAD::linearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u,
                                                                                const ocs2::PreComputation& preComputation) {
    ocs2::vector_t dx = ocs2::vector_t::Zero(2*pinInterface_.getModel().nv);
    tapedTimeStateInput_ << t, dx, u;
    flowJacobian_ = flowMapADInterfacePtr_->getJacobian(tapedTimeStateInput_, x);

    ocs2::VectorFunctionLinearApproximation approximation;
    approximation.dfdx = flowJacobian_.middleCols(1, dx.rows());
    approximation.dfdu = flowJacobian_.rightCols(u.rows());
    approximation.f = flowMapADInterfacePtr_->getFunctionValue(tapedTimeStateInput_, x);
    return approximation;
  }

  ocs2::VectorFunctionLinearApproximation SystemDynamicsAD::jumpMapLinearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x,
                                                                                     const ocs2::PreComputation& preComputation) {
    ocs2::vector_t dx = ocs2::vector_t::Zero(2*pinInterface_.getModel().nv);
    tapedTimeState_ << t, dx;
    jumpJacobian_ = jumpMapADInterfacePtr_->getJacobian(tapedTimeState_, x);

    ocs2::VectorFunctionLinearApproximation approximation;
    approximation.dfdx = jumpJacobian_.rightCols(dx.rows());
    approximation.dfdu.setZero(jumpJacobian_.rows(), 0);
    approximation.f = jumpMapADInterfacePtr_->getFunctionValue(tapedTimeState_, x);
    return approximation;
  }
}
