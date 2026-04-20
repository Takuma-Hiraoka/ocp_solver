#include "ocp_solver/ocp_interface.h"
#include "ocp_solver/system_dynamics_ad.h"
#include "ocp_solver/ocp_pre_computation.h"
#include "ocp_solver/gravity_compensation_initializer.h"

namespace ocp_solver {
  void OCPInterface::Initialize(const std::string& urdfFile, const std::vector<std::string> fixedJointNames, const std::vector<ContactCandidate>& contactCandidates, const ocs2::ModeSchedule& initModeSchedule, const ocs2::scalar_t& phaseTransitionIdleTime, const pinocchio::JointModelComposite& baseJointComposite) {
    pinocchio::ModelTpl<double> pinocchioModel;
    urdf::ModelInterfaceSharedPtr urdfModel;
    pinocchio_model_builder::buildModel(pinocchioModel, urdfFile, fixedJointNames, baseJointComposite, urdfModel);
    pinocchioInterfacePtr_.reset(new ocs2::PinocchioInterface(pinocchioModel, urdfModel));

    std::vector<std::string> jointNames;
    for (pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex)pinocchioModel.njoints; ++joint_id) {
      if (std::find(fixedJointNames.begin(), fixedJointNames.end(), pinocchioModel.names[joint_id]) == fixedJointNames.end()) jointNames.push_back(pinocchioModel.names[joint_id]);
    }
    std::unordered_map<std::string, size_t> jointIndexMap;
    for (size_t i = 0; i < jointNames.size(); ++i) {
      jointIndexMap[jointNames[i]] = i;
    }
    stateConverterPtr_.reset(new StateConverter<ocs2::scalar_t>(jointNames.size(), contactCandidates, jointIndexMap, baseJointComposite.nq()));
    stateConverterADPtr_.reset(new StateConverter<ocs2::ad_scalar_t>(jointNames.size(), contactCandidates, jointIndexMap, baseJointComposite.nq()));

    referenceManagerPtr_ = std::make_shared<SwitchedModelReferenceManager>(std::make_shared<ContactSchedule>(initModeSchedule, phaseTransitionIdleTime));

    initializerPtr_.reset(new GravityCompensationInitializer(*pinocchioInterfacePtr_, *referenceManagerPtr_, *stateConverterPtr_));

    problemPtr_.reset(new ocs2::OptimalControlProblem);

    std::unique_ptr<ocs2::SystemDynamicsBase> dynamicsPtr;
    const std::string modelName = "dynamics";
    dynamicsPtr.reset(new SystemDynamicsAD(*pinocchioInterfacePtr_, *stateConverterADPtr_, modelName, "build/cppad_autocode_gen"));
    problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

    problemPtr_->preComputationPtr.reset(new OCPPreComputation(*pinocchioInterfacePtr_, *stateConverterPtr_));
  }
}
