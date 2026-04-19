#pragma once
#include <ocs2_robotic_tools/common/RobotInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <pinocchio_model_builder/pinocchio_model_builder.h>
#include "ocp_solver/state_converter.h"

namespace ocp_solver {
  static pinocchio::JointModelComposite getBaseJointcomposite() {
    pinocchio::JointModelComposite baseJointComposite(2);
    baseJointComposite.addJoint(pinocchio::JointModelTranslation());
    baseJointComposite.addJoint(pinocchio::JointModelSphericalZYX());
    return baseJointComposite;
  }
  class OCPInterface : public ocs2::RobotInterface {
  public:
    ~OCPInterface() override = default;
    void Initialize(const std::string& urdfFile, const std::vector<std::string> fixedJointNames, const pinocchio::JointModelComposite& baseJointComposite=getBaseJointcomposite());

    const ocs2::PinocchioInterface& getPinocchioInterface() const { return *this->pinocchioInterfacePtr_; }
    const StateConverter<ocs2::scalar_t>& getStateConverter() const { return *stateConverterPtr_; }
  private:
    std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr_;
    std::unique_ptr<StateConverter<ocs2::scalar_t>> stateConverterPtr_;
  };
}
