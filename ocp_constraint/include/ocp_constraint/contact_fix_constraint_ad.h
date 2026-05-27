#pragma once

#include "ocp_constraint/position_constraint_ad.h"
#include <ocp_solver/solver/switched_model_reference_manager.h>

namespace ocp_constraint {

  class ContactFixConstraintAD final : public PositionConstraintAD {
  public:
    ContactFixConstraintAD(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                           const ocp_solver::PinocchioFrameDynamicsCppAd& frameDynamics,
                           size_t numConstraints,
                           PositionConstraintAD::Config config = PositionConstraintAD::Config());

    ~ContactFixConstraintAD() override = default;
    ContactFixConstraintAD* clone() const override { return new ContactFixConstraintAD(*this); }

    bool isActive(ocs2::scalar_t time) const override;

    const pinocchio::SE3 getTargetPose(ocs2::scalar_t time) const override;

  private:
    ContactFixConstraintAD(const ContactFixConstraintAD& rhs);
    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_;
  };

}
