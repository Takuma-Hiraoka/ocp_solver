#pragma once
#include <ocs2_robotic_tools/common/RobotInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <pinocchio_model_builder/pinocchio_model_builder.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_sqp/SqpSettings.h>
#include <ocs2_sqp/SqpMpc.h>
#include "ocp_solver/solver/state_converter.h"
#include "ocp_solver/contact_candidate.h"
#include "ocp_solver/solver/switched_model_reference_manager.h"
#include "ocp_solver/solver/ocp_sqp_mpc.h"

namespace ocp_solver {
  static pinocchio::JointModelComposite getBaseJointcompositeForAD() {
    pinocchio::JointModelComposite baseJointComposite(2);
    baseJointComposite.addJoint(pinocchio::JointModelTranslation());
    baseJointComposite.addJoint(pinocchio::JointModelSphericalZYX());
    return baseJointComposite;
  }
  class OCPInterface : public ocs2::RobotInterface {
  public:
    OCPInterface() {
      mpcSettings_.solutionTimeWindow_ = -1;
      mpcSettings_.coldStart_ = false;
      mpcSettings_.debugPrint_ = false;

      sqpSettings_.nThreads = 4;
      sqpSettings_.sqpIteration = 100;
      sqpSettings_.deltaTol = 1e-4;
      sqpSettings_.g_max = 1e-2;
      sqpSettings_.g_min = 1e-6;
      sqpSettings_.inequalityConstraintMu = 0.1;
      sqpSettings_.inequalityConstraintDelta = 5.0;
      sqpSettings_.projectStateInputEqualityConstraints = true;
      sqpSettings_.printSolverStatistics = false;
      sqpSettings_.printSolverStatus = false;
      sqpSettings_.printLinesearch = false;
      sqpSettings_.useFeedbackPolicy = false;
      sqpSettings_.integratorType = ocs2::SensitivityIntegratorType::EULER;
      sqpSettings_.threadPriority = 90;
    }
    ~OCPInterface() override = default;
    void initialize(const std::string& taskName, const std::string& urdfFile, const std::vector<std::string> fixedJointNames, const bool& useAD=false, const std::vector<ContactCandidate>& contactCandidates=std::vector<ContactCandidate>(), const pinocchio::JointModelComposite& baseJointComposite=pinocchio::JointModelFreeFlyer());
    std::shared_ptr<OcpSqpMpc> createSqpMpc();
    void addContactFrame(const std::vector<ContactCandidate>& contactCandidates, pinocchio::ModelTpl<double>& model);
    void createJointInfo(const std::vector<std::string> fixedJointNames, const pinocchio::JointModelComposite& baseJointComposite, const pinocchio::ModelTpl<double>& model, std::vector<std::string>& jointNames, std::unordered_map<std::string, size_t>& jointIndexMap);

    ocs2::PinocchioInterface& getPinocchioInterface() { return *this->pinocchioInterfacePtr_; }
    const ocs2::PinocchioInterface& getPinocchioInterface() const { return *this->pinocchioInterfacePtr_; }
    StateConverter<ocs2::scalar_t>& getStateConverter() { return *stateConverterPtr_; }
    const StateConverter<ocs2::scalar_t>& getStateConverter() const { return *stateConverterPtr_; }
    StateConverter<ocs2::ad_scalar_t>& getStateConverterAD() { return *stateConverterADPtr_; }
    const StateConverter<ocs2::ad_scalar_t>& getStateConverterAD() const { return *stateConverterADPtr_; }
    std::shared_ptr<SwitchedModelReferenceManager> getReferenceManagerPtr() { return referenceManagerPtr_; }
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
