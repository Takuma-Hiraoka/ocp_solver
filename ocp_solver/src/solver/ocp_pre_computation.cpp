#include <pinocchio/fwd.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <ocs2_core/misc/Numerics.h>

#include "ocp_solver/solver/ocp_pre_computation.h"
#include "ocp_solver/solver/dynamics_helper_functions.h"

namespace ocp_solver {

  OCPPreComputation::OCPPreComputation(ocs2::PinocchioInterface pinocchioInterface,
                                       const StateConverter<ocs2::scalar_t>& stateConverter)
    : pinocchioInterface_(std::move(pinocchioInterface)),
      stateConverterPtr_(&stateConverter) {}

  OCPPreComputation::OCPPreComputation(const OCPPreComputation& rhs)
    : pinocchioInterface_(rhs.pinocchioInterface_),
      stateConverterPtr_(rhs.stateConverterPtr_),
      q_(rhs.q_),
      v_(rhs.v_),
      a_(rhs.a_),
      input_(rhs.input_),
      baseAccelerationLinearApproximationValid_(rhs.baseAccelerationLinearApproximationValid_),
      baseAccelerationLinearApproximation_(rhs.baseAccelerationLinearApproximation_) {}

  OCPPreComputation* OCPPreComputation::clone() const {
    return new OCPPreComputation(*this);
  }

  void OCPPreComputation::updatePinocchioModelKinematics(const ocs2::vector_t& q, const ocs2::vector_t& v, const ocs2::vector_t& a) const {
    const pinocchio::Model& model = pinocchioInterface_.getModel();
    pinocchio::Data& data = pinocchioInterface_.getData();

    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::computeJointJacobiansTimeVariation(model, data, q, v);
    pinocchio::forwardKinematics(model, data, q, v, a);
    pinocchio::crba(model, data, q);
    pinocchio::nonLinearEffects(model, data, q, v);
    pinocchio::computeForwardKinematicsDerivatives(model, data, q, v, a);
  }

  const BaseAccelerationLinearApproximation& OCPPreComputation::getBaseAccelerationLinearApproximation() const {
    if (!baseAccelerationLinearApproximationValid_) {
      baseAccelerationLinearApproximation_ =
        computeBaseAccelerationLinearApproximation(q_, v_, a_, input_, pinocchioInterface_, *stateConverterPtr_);
      updatePinocchioModelKinematics(q_, v_, a_);
      baseAccelerationLinearApproximationValid_ = true;
    }
    return baseAccelerationLinearApproximation_;
  }

  void OCPPreComputation::request(ocs2::RequestSet request, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u) {
    if (!request.containsAny(ocs2::Request::Cost + ocs2::Request::Constraint + ocs2::Request::SoftConstraint)) {
      return;
    }

    StateConverter<ocs2::scalar_t>& sc = const_cast<StateConverter<ocs2::scalar_t>&>(*stateConverterPtr_);
    q_ = stateConverterPtr_->getGeneralizedCoordinates(x);
    v_ = stateConverterPtr_->getGeneralizedVelocities(x, u);
    a_ = computeGeneralizedAccelerations(x, u, pinocchioInterface_, sc);
    input_ = u;
    baseAccelerationLinearApproximationValid_ = false;
    updatePinocchioModelKinematics(q_, v_, a_);

  }

}
