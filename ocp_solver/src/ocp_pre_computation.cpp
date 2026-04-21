#include <pinocchio/fwd.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_core/misc/Numerics.h>

#include "ocp_solver/ocp_pre_computation.h"

namespace ocp_solver {

  OCPPreComputation::OCPPreComputation(ocs2::PinocchioInterface pinocchioInterface,
                                       const StateConverter<ocs2::scalar_t>& stateConverter)
    : pinocchioInterface_(std::move(pinocchioInterface)),
      stateConverterPtr_(&stateConverter) {
      R_world_to_contacts_.resize(stateConverterPtr_->getContactNum());
    for (size_t i = 0; i < stateConverterPtr_->getContactNum(); i++) {
      R_world_to_contacts_[i] = Eigen::Matrix<ocs2::scalar_t, 3, 3>::Identity();
    }
  }

  OCPPreComputation::OCPPreComputation(const OCPPreComputation& rhs)
    : pinocchioInterface_(rhs.pinocchioInterface_),
      stateConverterPtr_(rhs.stateConverterPtr_),
      R_world_to_contacts_(rhs.R_world_to_contacts_){}

  OCPPreComputation* OCPPreComputation::clone() const {
    return new OCPPreComputation(*this);
  }

  void OCPPreComputation::updatePinocchioModelKinematics(const ocs2::vector_t& q) {
    const pinocchio::Model& model = pinocchioInterface_.getModel();
    pinocchio::Data& data = pinocchioInterface_.getData();

    pinocchio::forwardKinematics(model, data, q);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);
  }

  void OCPPreComputation::request(ocs2::RequestSet request, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u) {
    if (!request.containsAny(ocs2::Request::Cost + ocs2::Request::Constraint + ocs2::Request::SoftConstraint)) {
      return;
    }

    updatePinocchioModelKinematics(stateConverterPtr_->getGeneralizedCoordinates(x));

    if (request.contains(ocs2::Request::Constraint)) {
      for (size_t i = 0; i < stateConverterPtr_->getContactNum(); i++) {
        pinocchio::FrameIndex frameID = pinocchioInterface_.getModel().getFrameId(stateConverterPtr_->getContactCandidates()[i].name);
        R_world_to_contacts_[i] = pinocchioInterface_.getData().oMf[frameID].rotation().inverse();
      }
    }
  }

}
