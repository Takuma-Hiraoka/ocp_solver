#include "ocp_solver/ocp_interface.h"
#include "ocp_solver/solver/ocp_sqp_solver.h"
#include "ocp_solver/solver/system_dynamics_ad.h"
#include "ocp_solver/solver/system_dynamics.h"
#include "ocp_solver/solver/ocp_pre_computation.h"
#include "ocp_solver/solver/gravity_compensation_initializer.h"
#include "ocp_solver/common/zero_wrench_constraint.h"
#include "ocp_solver/ocp_data/ocp_optimal_control_problem.h"

#include <stdexcept>

namespace ocp_solver {
  void OCPInterface::initialize(const std::string& taskName, const std::string& urdfFile, const std::vector<std::string> fixedJointNames, const bool& useAD, const std::vector<ContactCandidate>& contactCandidates, const pinocchio::JointModelComposite& baseJointComposite) {
    pinocchio::ModelTpl<double> pinocchioModel;
    urdf::ModelInterfaceSharedPtr urdfModel;
    pinocchio_model_builder::buildModel(pinocchioModel, urdfFile, fixedJointNames, baseJointComposite, urdfModel);

    pinocchioInterfacePtr_.reset(new ocs2::PinocchioInterface(pinocchioModel, urdfModel));

    std::vector<std::string> jointNames;
    std::unordered_map<std::string, size_t> jointIndexMap;
    createJointInfo(fixedJointNames, baseJointComposite, pinocchioInterfacePtr_->getModel(), jointNames, jointIndexMap);

    const std::vector<ContactCandidateInfo> contactCandidateInfo =
      createContactCandidateInfo(contactCandidates, pinocchioInterfacePtr_->getModel());

    stateConverterPtr_.reset(new StateConverter<ocs2::scalar_t>(jointNames.size(), contactCandidateInfo, jointIndexMap, baseJointComposite.nq(), baseJointComposite.nv()));
    if (useAD) {
      std::vector<ContactCandidateInfoTpl<ocs2::ad_scalar_t>> contactCandidateInfoAD;
      contactCandidateInfoAD.reserve(contactCandidateInfo.size());
      for (const ContactCandidateInfo& candidate : contactCandidateInfo) {
        ContactCandidateInfoTpl<ocs2::ad_scalar_t> candidateAD;
        candidateAD.index = candidate.index;
        candidateAD.frameName = candidate.frameName;
        candidateAD.parentJointIndex = candidate.parentJointIndex;
        candidateAD.localFramePose = pinocchio::SE3Tpl<ocs2::ad_scalar_t>(
          candidate.localFramePose.rotation().cast<ocs2::ad_scalar_t>(),
          candidate.localFramePose.translation().cast<ocs2::ad_scalar_t>());
        candidateAD.localPoseInLocalFrame = pinocchio::SE3Tpl<ocs2::ad_scalar_t>(
          candidate.localPoseInLocalFrame.rotation().cast<ocs2::ad_scalar_t>(),
          candidate.localPoseInLocalFrame.translation().cast<ocs2::ad_scalar_t>());
        candidateAD.localPose = pinocchio::SE3Tpl<ocs2::ad_scalar_t>(
          candidate.localPose.rotation().cast<ocs2::ad_scalar_t>(),
          candidate.localPose.translation().cast<ocs2::ad_scalar_t>());
        candidateAD.searchContactPoint = candidate.searchContactPoint;
        candidateAD.alignContactFrameWithMeshNormal = false;
        candidateAD.meshVerticesInLocalFrame = candidate.meshVerticesInLocalFrame;
        candidateAD.meshNormalsInLocalFrame = candidate.meshNormalsInLocalFrame;
        candidateAD.contactPointStateIndex = candidate.contactPointStateIndex;
        contactCandidateInfoAD.push_back(candidateAD);
      }
      stateConverterADPtr_.reset(new StateConverter<ocs2::ad_scalar_t>(jointNames.size(), contactCandidateInfoAD, jointIndexMap, baseJointComposite.nq(), baseJointComposite.nv()));
    }

    referenceManagerPtr_ = std::make_shared<SwitchedModelReferenceManager>();

    initializerPtr_.reset(new GravityCompensationInitializer(*pinocchioInterfacePtr_, *referenceManagerPtr_, *stateConverterPtr_));

    problemPtr_.reset(new OptimalControlProblem(stateConverterPtr_->getStateVariableDim(), stateConverterPtr_->getInputDim()));

    std::unique_ptr<ocs2::SystemDynamicsBase> dynamicsPtr;
    const std::string modelName = taskName + "_dynamics";
    if (useAD) dynamicsPtr.reset(new SystemDynamicsAD(*pinocchioInterfacePtr_, *stateConverterADPtr_, modelName, "build/cppad_autocode_gen"));
    else dynamicsPtr.reset(new SystemDynamics(*pinocchioInterfacePtr_, *stateConverterPtr_));
    problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

    for (size_t i=0; i<contactCandidates.size(); i++) {
      problemPtr_->equalityConstraintPtr->add(contactCandidates[i].frameName + "_zero_wrench", std::unique_ptr<ocs2::StateInputConstraint>(std::make_unique<ocp_solver::ZeroWrenchConstraint>(*referenceManagerPtr_, i, *stateConverterPtr_)));
    }

    problemPtr_->preComputationPtr.reset(new OCPPreComputation(*pinocchioInterfacePtr_, *stateConverterPtr_));
  }

  std::shared_ptr<OcpSqpMpc> OCPInterface::createSqpMpc() {
    std::shared_ptr<OcpSqpMpc> mpc = std::make_shared<OcpSqpMpc>(mpcSettings_, sqpSettings_, *problemPtr_, *initializerPtr_);
    mpc->getSolverPtr()->setReferenceManager(referenceManagerPtr_);
    return mpc;
  }

  std::unique_ptr<PoseOptimizer> OCPInterface::createPoseOptimizer() {
    problemPtr_->targetTrajectoriesPtr = &referenceManagerPtr_->getTargetTrajectories();
    return std::make_unique<PoseOptimizer>(sqpSettings_, *problemPtr_, *stateConverterPtr_, *pinocchioInterfacePtr_);
  }

  std::vector<ContactCandidateInfo> OCPInterface::createContactCandidateInfo(const std::vector<ContactCandidate>& contactCandidates,
                                                                             const pinocchio::ModelTpl<double>& model) const {
    std::vector<ContactCandidateInfo> contactCandidateInfo;
    contactCandidateInfo.reserve(contactCandidates.size());
    size_t contactPointStateIndex = 0;
    for (size_t i = 0; i < contactCandidates.size(); ++i) {
      ContactCandidateInfo candidateInfo;
      candidateInfo.index = i;
      candidateInfo.frameName = contactCandidates[i].frameName;
      const pinocchio::JointIndex jointIndex = model.getJointId(contactCandidates[i].parentJointName);
      if (jointIndex < static_cast<pinocchio::JointIndex>(model.njoints)
          && model.names[jointIndex] == contactCandidates[i].parentJointName) {
        candidateInfo.parentJointIndex = jointIndex;
        candidateInfo.localFramePose = pinocchio::SE3::Identity();
        candidateInfo.localPoseInLocalFrame = contactCandidates[i].localPose;
        candidateInfo.localPose = contactCandidates[i].localPose;
      } else {
        const pinocchio::FrameIndex frameIndex = model.getFrameId(contactCandidates[i].parentJointName);
        if (frameIndex >= model.frames.size()) {
          throw std::runtime_error("Contact candidate parent joint or frame not found: "
                                   + contactCandidates[i].parentJointName);
        }
        const pinocchio::Frame& frame = model.frames[frameIndex];
        candidateInfo.parentJointIndex = frame.parentJoint;
        candidateInfo.localFramePose = frame.placement;
        candidateInfo.localPoseInLocalFrame = contactCandidates[i].localPose;
        candidateInfo.localPose = frame.placement * contactCandidates[i].localPose;
      }
      candidateInfo.searchContactPoint = contactCandidates[i].searchContactPoint;
      candidateInfo.alignContactFrameWithMeshNormal = contactCandidates[i].alignContactFrameWithMeshNormal;
      candidateInfo.meshVerticesInLocalFrame = contactCandidates[i].meshVerticesInLocalFrame;
      candidateInfo.meshNormalsInLocalFrame = contactCandidates[i].meshNormalsInLocalFrame;
      if (candidateInfo.searchContactPoint) {
        candidateInfo.contactPointStateIndex = contactPointStateIndex++;
      }
      contactCandidateInfo.push_back(candidateInfo);
    }
    return contactCandidateInfo;
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
