#pragma once
#include <ocs2_robotic_tools/common/RobotInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <pinocchio_model_builder/pinocchio_model_builder.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_sqp/SqpSettings.h>
#include "ocp_solver/state_converter.h"
#include "ocp_solver/contact_candidate.h"
#include "ocp_solver/switched_model_reference_manager.h"

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
    void Initialize(const std::string& urdfFile, const std::vector<std::string> fixedJointNames, const std::vector<ContactCandidate>& contactCandidates=std::vector<ContactCandidate>(), const ocs2::ModeSchedule& initModeSchedule=ocs2::ModeSchedule(), const ocs2::scalar_t& phaseTransitionIdleTime=0.5, const pinocchio::JointModelComposite& baseJointComposite=getBaseJointcomposite());

    const ocs2::PinocchioInterface& getPinocchioInterface() const { return *this->pinocchioInterfacePtr_; }
    const StateConverter<ocs2::scalar_t>& getStateConverter() const { return *stateConverterPtr_; }
    const StateConverter<ocs2::ad_scalar_t>& getStateConverterAD() const { return *stateConverterADPtr_; }
    std::shared_ptr<ocs2::ReferenceManagerInterface> getReferenceManagerPtr() const override { return referenceManagerPtr_; }
    const ocs2::OptimalControlProblem& getOptimalControlProblem() const override { return *this->problemPtr_; }
    const ocs2::Initializer& getInitializer() const override { return *this->initializerPtr_; }

    const ocs2::mpc::Settings& mpcSettings() const { return mpcSettings_; }
    ocs2::mpc::Settings& mpcSettings() { return mpcSettings_; }
    const ocs2::sqp::Settings& sqpSettings() const { return sqpSettings_; }
    ocs2::sqp::Settings& sqpSettings() { return sqpSettings_; }
  private:
    std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr_;
    std::unique_ptr<StateConverter<ocs2::scalar_t>> stateConverterPtr_;
    std::unique_ptr<StateConverter<ocs2::ad_scalar_t>> stateConverterADPtr_;
    std::shared_ptr<SwitchedModelReferenceManager> referenceManagerPtr_;
    std::unique_ptr<ocs2::OptimalControlProblem> problemPtr_;
    std::unique_ptr<ocs2::Initializer> initializerPtr_;

    ocs2::mpc::Settings mpcSettings_;
    ocs2::sqp::Settings sqpSettings_;
  };
}
