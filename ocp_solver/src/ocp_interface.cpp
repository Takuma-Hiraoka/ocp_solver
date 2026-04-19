#include "ocp_solver/ocp_interface.h"

namespace ocp_solver {
  void OCPInterface::Initialize(const std::string& urdfFile, const std::vector<std::string> fixedJointNames, const pinocchio::JointModelComposite& baseJointComposite) {
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
    stateConverterPtr_.reset(new StateConverter<ocs2::scalar_t>(jointNames.size(), 0/*TODO*/, jointIndexMap));
  }
}
