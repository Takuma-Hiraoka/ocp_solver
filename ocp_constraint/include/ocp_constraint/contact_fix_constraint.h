#pragma once

#include "ocp_constraint/position_constraint.h"
#include <ocp_solver/solver/switched_model_reference_manager.h>

namespace ocp_constraint {

  class ContactFixConstraint final : public PositionConstraint {
  public:
    ContactFixConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                         const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                         size_t numConstraints,
                         PositionConstraint::Config config = PositionConstraint::Config());

    ~ContactFixConstraint() override = default;
    ContactFixConstraint* clone() const override { return new ContactFixConstraint(*this); }

    bool isActive(ocs2::scalar_t time) const override;

    const pinocchio::SE3 getTargetPose(ocs2::scalar_t time) const override;

  private:
    ContactFixConstraint(const ContactFixConstraint& rhs);
    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_;
  };
}
