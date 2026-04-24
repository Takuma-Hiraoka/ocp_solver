#include "ocp_solver/ocp_interface.h"
#include "ocp_solver/system_dynamics_ad.h"
#include "ocp_solver/ocp_pre_computation.h"
#include "ocp_solver/gravity_compensation_initializer.h"

namespace ocp_solver {
  void OCPInterface::initialize(const std::string& taskName, const std::string& urdfFile, const std::vector<std::string> fixedJointNames, const std::vector<ContactCandidate>& contactCandidates, const ocs2::ModeSchedule& initModeSchedule, const ocs2::scalar_t& phaseTransitionIdleTime, const pinocchio::JointModelComposite& baseJointComposite) {
    pinocchio::ModelTpl<double> pinocchioModel;
    urdf::ModelInterfaceSharedPtr urdfModel;
    pinocchio_model_builder::buildModel(pinocchioModel, urdfFile, fixedJointNames, baseJointComposite, urdfModel);
    addContactFrame(contactCandidates, pinocchioModel);

    pinocchioInterfacePtr_.reset(new ocs2::PinocchioInterface(pinocchioModel, urdfModel));

    std::vector<std::string> jointNames;
    std::unordered_map<std::string, size_t> jointIndexMap;
    createJointInfo(fixedJointNames, baseJointComposite, pinocchioInterfacePtr_->getModel(), jointNames, jointIndexMap);

    stateConverterPtr_.reset(new StateConverter<ocs2::scalar_t>(jointNames.size(), contactCandidates, jointIndexMap, baseJointComposite.nq()));
    stateConverterADPtr_.reset(new StateConverter<ocs2::ad_scalar_t>(jointNames.size(), contactCandidates, jointIndexMap, baseJointComposite.nq()));

    referenceManagerPtr_ = std::make_shared<SwitchedModelReferenceManager>(std::make_shared<ContactSchedule>(initModeSchedule, phaseTransitionIdleTime));

    mappingPtr_.reset(new OCPPinocchioMapping(*stateConverterPtr_));

    initializerPtr_.reset(new GravityCompensationInitializer(*pinocchioInterfacePtr_, *referenceManagerPtr_, *stateConverterPtr_));

    problemPtr_.reset(new ocs2::OptimalControlProblem);

    std::unique_ptr<ocs2::SystemDynamicsBase> dynamicsPtr;
    const std::string modelName = taskName + "_dynamics";
    dynamicsPtr.reset(new SystemDynamicsAD(*pinocchioInterfacePtr_, *stateConverterADPtr_, modelName, "build/cppad_autocode_gen"));
    problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

    problemPtr_->preComputationPtr.reset(new OCPPreComputation(*pinocchioInterfacePtr_, *stateConverterPtr_));
  }

  std::shared_ptr<ocs2::SqpMpc> OCPInterface::createSqpMpc() {
    std::shared_ptr<ocs2::SqpMpc> mpc = std::make_shared<ocs2::SqpMpc>(mpcSettings_, sqpSettings_, *problemPtr_, *initializerPtr_);
    mpc->getSolverPtr()->setReferenceManager(referenceManagerPtr_);
    return mpc;
  }

  void OCPInterface::addContactFrame(const std::vector<ContactCandidate>& contactCandidates, pinocchio::ModelTpl<double>& model) {
    for (ContactCandidate candidate : contactCandidates) {
      pinocchio::Frame contactCenterFrame(candidate.name, model.getJointId(candidate.parentJointName), model.getFrameId(candidate.parentJointName), candidate.localPose, pinocchio::FIXED_JOINT);
      model.addFrame(contactCenterFrame);
    }
  }

  void OCPInterface::createJointInfo(const std::vector<std::string> fixedJointNames, const pinocchio::JointModelComposite& baseJointComposite, const pinocchio::ModelTpl<double>& model, std::vector<std::string>& jointNames, std::unordered_map<std::string, size_t>& jointIndexMap) {
    size_t joint_id_offset = (baseJointComposite.nq() == 0) ? 1 : 2; // universe, root_joint
    for (pinocchio::JointIndex joint_id = joint_id_offset; joint_id < (pinocchio::JointIndex)model.njoints; ++joint_id) {
      if (std::find(fixedJointNames.begin(), fixedJointNames.end(), model.names[joint_id]) == fixedJointNames.end()) jointNames.push_back(model.names[joint_id]);
    }
    for (size_t i = 0; i < jointNames.size(); ++i) {
      jointIndexMap[jointNames[i]] = i;
    }
  }

}
