#include "ocp_solver/ocp_interface.h"

namespace ocp_solver {
  void OCPInterface::Initialize(const std::string& urdfFile, const std::vector<std::string> fixedJointNames, const pinocchio::JointModelComposite& baseJointComposite) {
    pinocchio::ModelTpl<double> pinocchioModel;
    urdf::ModelInterfaceSharedPtr urdfModel;
    pinocchio_model_builder::buildModel(pinocchioModel, urdfFile, fixedJointNames, baseJointComposite, urdfModel);
    pinocchioInterfacePtr_.reset(new ocs2::PinocchioInterface(pinocchioModel, urdfModel));
  }
}
