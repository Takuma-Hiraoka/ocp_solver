#include "ocp_solver/solver/system_dynamics.h"
#include "ocp_solver/solver/dynamics_helper_functions.h"
#include "ocp_solver/solver/ocp_pre_computation.h"

namespace ocp_solver {
  SystemDynamics::SystemDynamics(const ocs2::PinocchioInterface& pinocchioInterface,
                                 StateConverter<ocs2::scalar_t>& stateConverter)
    : SystemDynamicsBase(),
      pinInterface_(pinocchioInterface),
      stateConverter_(stateConverter) {}

  ocs2::vector_t SystemDynamics::computeFlowMap(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::PreComputation& preComputation) {
    return computeStateDerivative<ocs2::scalar_t>(x, u, pinInterface_, stateConverter_);
  }

  ocs2::vector_t SystemDynamics::computeJumpMap(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::PreComputation& preComputation) {
    return x;
  }

  ocs2::VectorFunctionLinearApproximation SystemDynamics::linearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u,
                                                                              const ocs2::PreComputation& preComputation) {

    const size_t tangentDim = stateConverter_.getTangentDim();
    const size_t jointDim = stateConverter_.getJointDim();
    const ocs2::vector_t q = stateConverter_.getGeneralizedCoordinates(x);
    const ocs2::vector_t v = stateConverter_.getGeneralizedVelocities(x, u);
    const ocs2::vector_t qddJoints = stateConverter_.getJointAccelerations(u);

    ocs2::VectorFunctionLinearApproximation approximation;
    approximation.dfdx.setZero(stateConverter_.getStateVariableDim(), stateConverter_.getStateVariableDim());
    approximation.dfdx.topRightCorner(tangentDim, tangentDim) = ocs2::matrix_t::Identity(tangentDim, tangentDim);
    approximation.dfdu.setZero(stateConverter_.getStateVariableDim(), u.rows());
    approximation.dfdu.bottomRightCorner(jointDim, jointDim) = ocs2::matrix_t::Identity(jointDim, jointDim);
    approximation.f = computeStateDerivative<ocs2::scalar_t>(x, u, pinInterface_, stateConverter_);

    if (stateConverter_.getBaseVDim() == 6) {
      ocs2::vector_t qdd(tangentDim);
      qdd.head(6) = approximation.f.segment(tangentDim, 6);
      qdd.tail(jointDim) = qddJoints;

      const auto* ocpPreComputation = dynamic_cast<const OCPPreComputation*>(&preComputation);
      const BaseAccelerationLinearApproximation baseAccelerationApprox =
        (ocpPreComputation != nullptr)
          ? ocpPreComputation->getBaseAccelerationLinearApproximation()
          : computeBaseAccelerationLinearApproximation(q, v, qdd, u, pinInterface_, stateConverter_);
      approximation.dfdx.block(tangentDim, 0, 6, tangentDim) = baseAccelerationApprox.dfdq;
      approximation.dfdx.block(tangentDim, tangentDim, 6, tangentDim) = baseAccelerationApprox.dfdv;
      approximation.dfdu.block(tangentDim, 0, 6, u.rows()) = baseAccelerationApprox.dfdu;
    }

    return approximation;
  }

  ocs2::VectorFunctionLinearApproximation SystemDynamics::jumpMapLinearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x,
                                                                                     const ocs2::PreComputation& preComputation) {
    ocs2::VectorFunctionLinearApproximation approximation;
    approximation.dfdx = ocs2::matrix_t::Identity(stateConverter_.getStateVariableDim(), stateConverter_.getStateVariableDim());
    approximation.dfdu.setZero(x.rows(), 0);
    approximation.f = x;
    return approximation;
  }
}
