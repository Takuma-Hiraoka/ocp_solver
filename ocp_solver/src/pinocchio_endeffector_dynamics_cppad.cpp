#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "ocp_solver/pinocchio_endeffector_dynamics_cppad.h"
#include "ocp_solver/dynamics_helper_functions_ad.h"

#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace {

  void defaultUpdatePinocchioInterface(const ocs2::ad_vector_t&, ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>&) {}

}  // unnamed namespace

namespace ocp_solver {

  PinocchioEndEffectorDynamicsCppAd::PinocchioEndEffectorDynamicsCppAd(const ocs2::PinocchioInterface& pinocchioInterface,
                                                                       StateConverter<ocs2::ad_scalar_t>& stateConverter,
                                                                       std::vector<std::string> endEffectorIds,
                                                                       const std::string& modelName,
                                                                       const std::string& modelFolder,
                                                                       bool recompileLibraries,
                                                                       bool verbose)

  : PinocchioEndEffectorDynamicsCppAd(pinocchioInterface,
                                      stateConverter,
                                      std::move(endEffectorIds),
                                      &defaultUpdatePinocchioInterface,
                                      modelName,
                                      modelFolder,
                                      recompileLibraries,
                                      verbose) {}

  PinocchioEndEffectorDynamicsCppAd::PinocchioEndEffectorDynamicsCppAd(const ocs2::PinocchioInterface& pinocchioInterface,
                                                                       StateConverter<ocs2::ad_scalar_t>& stateConverter,
                                                                       std::vector<std::string> endEffectorIds,
                                                                       update_pinocchio_interface_callback updateCallback,
                                                                       const std::string& modelName,
                                                                       const std::string& modelFolder,
                                                                       bool recompileLibraries,
                                                                       bool verbose)
  : endEffectorIds_(std::move(endEffectorIds)), pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()), mappingPtr_(&stateConverter) {
    for (const auto& bodyName : endEffectorIds_) {
      endEffectorFrameIds_.push_back(pinocchioInterface.getModel().getFrameId(bodyName));
    }

    size_t stateDim = mappingPtr_->getStateDim();
    size_t inputDim = mappingPtr_->getInputDim();

    // position function
    auto positionFunc = [&, this](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
                          updateCallback(x, pinocchioInterfaceCppAd_);
                          y = getPositionCppAd(x);
                        };
    positionCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(positionFunc, stateDim, modelName + "_position", modelFolder));

    // velocity function
    auto velocityFunc = [&, this](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
                          const ocs2::ad_vector_t state = x.head(stateDim);
                          const ocs2::ad_vector_t input = x.tail(inputDim);
                          updateCallback(state, pinocchioInterfaceCppAd_);
                          y = getVelocityCppAd(state, input);
                        };
    velocityCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(velocityFunc, stateDim + inputDim, modelName + "_velocity", modelFolder));

    // orientation function
    auto orientationFunc = [&, this](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
                             updateCallback(x, pinocchioInterfaceCppAd_);
                             y = getOrientationCppAd(x);
                           };
    orientationCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(orientationFunc, stateDim, modelName + "_orientation", modelFolder));

    // orientation function
    auto orientationErrorFunc = [&, this](const ocs2::ad_vector_t& x, const ocs2::ad_vector_t& params, ocs2::ad_vector_t& y) {
                                  updateCallback(x, pinocchioInterfaceCppAd_);
                                  y = getOrientationErrorCppAd(x, params);
                                };
    orientationErrorCppAdInterfacePtr_.reset(
                                             new ocs2::CppAdInterface(orientationErrorFunc, stateDim, 4 * endEffectorFrameIds_.size(), modelName + "_orientationError", modelFolder));

    // velocity function
    auto angularVelocityFunc = [&, this](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
                                 const ocs2::ad_vector_t state = x.head(stateDim);
                                 const ocs2::ad_vector_t input = x.tail(inputDim);
                                 updateCallback(state, pinocchioInterfaceCppAd_);
                                 y = getAngularVelocityCppAd(state, input);
                               };
    angularVelocityCppAdInterfacePtr_.reset(
                                            new ocs2::CppAdInterface(angularVelocityFunc, stateDim + inputDim, modelName + "_angular_velocity", modelFolder));

    // twist function
    auto twistFunc = [&, this](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
                       const ocs2::ad_vector_t state = x.head(stateDim);
                       const ocs2::ad_vector_t input = x.tail(inputDim);
                       updateCallback(state, pinocchioInterfaceCppAd_);
                       y = getTwistCppAd(state, input);
                     };
    twistCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(twistFunc, stateDim + inputDim, modelName + "_twist", modelFolder));

    // linear acceleration function
    auto linearAccelerationFunc = [&, this](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
                                    const ocs2::ad_vector_t state = x.head(stateDim);
                                    const ocs2::ad_vector_t input = x.tail(inputDim);
                                    updateCallback(state, pinocchioInterfaceCppAd_);
                                    y = getLinearAccelerationCppAd(state, input);
                                  };
    linearAccelerationCppAdInterfacePtr_.reset(
                                               new ocs2::CppAdInterface(linearAccelerationFunc, stateDim + inputDim, modelName + "_linear_acceleration", modelFolder));

    // velocity function
    auto angularAccelerationFunc = [&, this](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
                                     const ocs2::ad_vector_t state = x.head(stateDim);
                                     const ocs2::ad_vector_t input = x.tail(inputDim);
                                     updateCallback(state, pinocchioInterfaceCppAd_);
                                     y = getAngularAccelerationCppAd(state, input);
                                   };
    angularAccelerationCppAdInterfacePtr_.reset(
                                                new ocs2::CppAdInterface(angularAccelerationFunc, stateDim + inputDim, modelName + "_angular_acceleration", modelFolder));

    // twist function
    auto accelerationsFunc = [&, this](const ocs2::ad_vector_t& x, ocs2::ad_vector_t& y) {
                               const ocs2::ad_vector_t state = x.head(stateDim);
                               const ocs2::ad_vector_t input = x.tail(inputDim);
                               updateCallback(state, pinocchioInterfaceCppAd_);
                               y = getAccelerationsCppAd(state, input);
                             };
    accelerationsCppAdInterfacePtr_.reset(
                                          new ocs2::CppAdInterface(accelerationsFunc, stateDim + inputDim, modelName + "_accelerations", modelFolder));

    if (recompileLibraries) {
      positionCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      velocityCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      orientationCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::Zero, verbose);
      orientationErrorCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      angularVelocityCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      twistCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      linearAccelerationCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      angularAccelerationCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      accelerationsCppAdInterfacePtr_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
    } else {
      positionCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      velocityCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      orientationCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::Zero, verbose);
      orientationErrorCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      angularVelocityCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      twistCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      linearAccelerationCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      angularAccelerationCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
      accelerationsCppAdInterfacePtr_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
    }
  }

  PinocchioEndEffectorDynamicsCppAd::PinocchioEndEffectorDynamicsCppAd(const PinocchioEndEffectorDynamicsCppAd& rhs)
    : EndEffectorKinematics<ocs2::scalar_t>(rhs),
      positionCppAdInterfacePtr_(new ocs2::CppAdInterface(*rhs.positionCppAdInterfacePtr_)),
      velocityCppAdInterfacePtr_(new ocs2::CppAdInterface(*rhs.velocityCppAdInterfacePtr_)),
      orientationCppAdInterfacePtr_(new ocs2::CppAdInterface(*rhs.orientationCppAdInterfacePtr_)),
      orientationErrorCppAdInterfacePtr_(new ocs2::CppAdInterface(*rhs.orientationErrorCppAdInterfacePtr_)),
      angularVelocityCppAdInterfacePtr_(new ocs2::CppAdInterface(*rhs.angularVelocityCppAdInterfacePtr_)),
      twistCppAdInterfacePtr_(new ocs2::CppAdInterface(*rhs.twistCppAdInterfacePtr_)),
      linearAccelerationCppAdInterfacePtr_(new ocs2::CppAdInterface(*rhs.linearAccelerationCppAdInterfacePtr_)),
      angularAccelerationCppAdInterfacePtr_(new ocs2::CppAdInterface(*rhs.angularAccelerationCppAdInterfacePtr_)),
      accelerationsCppAdInterfacePtr_(new ocs2::CppAdInterface(*rhs.accelerationsCppAdInterfacePtr_)),
      endEffectorIds_(rhs.endEffectorIds_),
      endEffectorFrameIds_(rhs.endEffectorFrameIds_),
      pinocchioInterfaceCppAd_(rhs.pinocchioInterfaceCppAd_),
      mappingPtr_(rhs.mappingPtr_) {}

  PinocchioEndEffectorDynamicsCppAd* PinocchioEndEffectorDynamicsCppAd::clone() const {
    return new PinocchioEndEffectorDynamicsCppAd(*this);
  }

  const std::vector<std::string>& PinocchioEndEffectorDynamicsCppAd::getIds() const {
    return endEffectorIds_;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getPositionCppAd(const ocs2::ad_vector_t& state) {
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = mappingPtr_->getGeneralizedCoordinates(state);

    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);

    ocs2::ad_vector_t positions(3 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      positions.segment<3>(3 * i) = data.oMf[frameId].translation();
    }
    return positions;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getPosition(const vector_t& state) const -> std::vector<vector3_t> {
    const vector_t positionValues = positionCppAdInterfacePtr_->getFunctionValue(state);

    std::vector<vector3_t> positions;
    positions.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      positions.emplace_back(positionValues.segment<3>(3 * i));
    }
    return positions;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getPositionLinearApproximation(
                                                                                                                         const vector_t& state) const {
    const vector_t positionValues = positionCppAdInterfacePtr_->getFunctionValue(state);
    const ocs2::matrix_t positionJacobian = positionCppAdInterfacePtr_->getJacobian(state);

    std::vector<ocs2::VectorFunctionLinearApproximation> positions;
    positions.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation pos;
      pos.f = positionValues.segment<3>(3 * i);
      pos.dfdx = positionJacobian.block(3 * i, 0, 3, state.rows());
      positions.emplace_back(std::move(pos));
    }
    return positions;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getVelocityCppAd(const ocs2::ad_vector_t& state, const ocs2::ad_vector_t& input) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = mappingPtr_->getGeneralizedCoordinates(state);
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(state, input);

    pinocchio::forwardKinematics(model, data, q, v);

    ocs2::ad_vector_t velocities(3 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      velocities.segment<3>(3 * i) = pinocchio::getFrameVelocity(model, data, frameId, rf).linear();
    }
    return velocities;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getVelocity(const vector_t& state, const vector_t& input) const -> std::vector<vector3_t> {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t velocityValues = velocityCppAdInterfacePtr_->getFunctionValue(stateInput);

    std::vector<vector3_t> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      velocities.emplace_back(velocityValues.segment<3>(3 * i));
    }
    return velocities;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getVelocityLinearApproximation(
                                                                                                                         const vector_t& state, const vector_t& input) const {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t velocityValues = velocityCppAdInterfacePtr_->getFunctionValue(stateInput);
    const ocs2::matrix_t velocityJacobian = velocityCppAdInterfacePtr_->getJacobian(stateInput);

    std::vector<ocs2::VectorFunctionLinearApproximation> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation vel;
      vel.f = velocityValues.segment<3>(3 * i);
      vel.dfdx = velocityJacobian.block(3 * i, 0, 3, state.rows());
      vel.dfdu = velocityJacobian.block(3 * i, state.rows(), 3, input.rows());
      velocities.emplace_back(std::move(vel));
    }
    return velocities;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getOrientation(const vector_t& state) const -> std::vector<quaternion_t> {
    const vector_t orientationValues = orientationCppAdInterfacePtr_->getFunctionValue(state);

    std::vector<quaternion_t> orientations;
    orientations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      orientations.emplace_back(quaternion_t(orientationValues.segment<4>(4 * i)));
    }
    return orientations;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getOrientationCppAd(const ocs2::ad_vector_t& state) {
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = mappingPtr_->getGeneralizedCoordinates(state);

    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);

    ocs2::ad_vector_t orientations(4 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      orientations.segment<4>(4 * i) = ocs2::matrixToQuaternion(data.oMf[endEffectorFrameIds_[i]].rotation()).coeffs();
    }
    return orientations;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getOrientationError(
                                                              const vector_t& state, const std::vector<quaternion_t>& referenceOrientations) const -> std::vector<vector3_t> {
    vector_t params(4 * endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      params.segment<4>(i * 4) = referenceOrientations[i].coeffs();
    }

    const vector_t errorValues = orientationErrorCppAdInterfacePtr_->getFunctionValue(state, params);

    std::vector<vector3_t> errors;
    errors.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      errors.emplace_back(errorValues.segment<3>(3 * i));
    }
    return errors;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getOrientationErrorLinearApproximation(
                                                                                                                                 const vector_t& state, const std::vector<quaternion_t>& referenceOrientations) const {
    vector_t params(4 * endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      params.segment<4>(i * 4) = referenceOrientations[i].coeffs();
    }

    const vector_t errorValues = orientationErrorCppAdInterfacePtr_->getFunctionValue(state, params);
    const ocs2::matrix_t errorJacobian = orientationErrorCppAdInterfacePtr_->getJacobian(state, params);

    std::vector<ocs2::VectorFunctionLinearApproximation> errors;
    errors.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation err;
      err.f = errorValues.segment<3>(3 * i);
      err.dfdx = errorJacobian.block(3 * i, 0, 3, state.rows());
      errors.emplace_back(std::move(err));
    }
    return errors;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getOrientationErrorCppAd(const ocs2::ad_vector_t& state, const ocs2::ad_vector_t& params) {
    using ad_quaternion_t = Eigen::Quaternion<ocs2::ad_scalar_t>;

    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = mappingPtr_->getGeneralizedCoordinates(state);

    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);

    ocs2::ad_vector_t errors(3 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      const ad_quaternion_t eeOrientation = ocs2::matrixToQuaternion(data.oMf[frameId].rotation());
      ad_quaternion_t eeReferenceOrientation;
      eeReferenceOrientation.coeffs() = params.segment<4>(4 * i);
      errors.segment<3>(3 * i) = ocs2::quaternionDistance(eeOrientation, eeReferenceOrientation);
    }
    return errors;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getAngularVelocityCppAd(const ocs2::ad_vector_t& state, const ocs2::ad_vector_t& input) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = mappingPtr_->getGeneralizedCoordinates(state);
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(state, input);

    pinocchio::forwardKinematics(model, data, q, v);

    ocs2::ad_vector_t angularVelocities(3 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      angularVelocities.segment<3>(3 * i) = pinocchio::getFrameVelocity(model, data, frameId, rf).angular();
    }
    return angularVelocities;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getAngularVelocity(const vector_t& state, const vector_t& input) const -> std::vector<vector3_t> {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t velocityValues = angularVelocityCppAdInterfacePtr_->getFunctionValue(stateInput);

    std::vector<vector3_t> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      velocities.emplace_back(velocityValues.segment<3>(3 * i));
    }
    return velocities;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getAngularVelocityLinearApproximation(
                                                                                                                                const vector_t& state, const vector_t& input) const {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t velocityValues = angularVelocityCppAdInterfacePtr_->getFunctionValue(stateInput);
    const ocs2::matrix_t velocityJacobian = angularVelocityCppAdInterfacePtr_->getJacobian(stateInput);

    std::vector<ocs2::VectorFunctionLinearApproximation> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation vel;
      vel.f = velocityValues.segment<3>(3 * i);
      vel.dfdx = velocityJacobian.block(3 * i, 0, 3, state.rows());
      vel.dfdu = velocityJacobian.block(3 * i, state.rows(), 3, input.rows());
      velocities.emplace_back(std::move(vel));
    }
    return velocities;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getTwistCppAd(const ocs2::ad_vector_t& state, const ocs2::ad_vector_t& input) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = mappingPtr_->getGeneralizedCoordinates(state);
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(state, input);

    pinocchio::forwardKinematics(model, data, q, v);

    ocs2::ad_vector_t twists(6 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      auto motion = pinocchio::getFrameVelocity(model, data, frameId, rf);
      ocs2::ad_vector_t currTwist(6);
      currTwist.head(3) = motion.linear();
      currTwist.tail(3) = motion.angular();
      twists.segment<6>(6 * i) = currTwist;
    }
    return twists;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getTwist(const vector_t& state, const vector_t& input) const -> std::vector<vector6_t> {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t velocityValues = twistCppAdInterfacePtr_->getFunctionValue(stateInput);

    std::vector<vector6_t> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      velocities.emplace_back(velocityValues.segment<6>(6 * i));
    }
    return velocities;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getTwistLinearApproximation(const vector_t& state,
                                                                                                                      const vector_t& input) const {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t velocityValues = twistCppAdInterfacePtr_->getFunctionValue(stateInput);
    const ocs2::matrix_t velocityJacobian = twistCppAdInterfacePtr_->getJacobian(stateInput);

    std::vector<ocs2::VectorFunctionLinearApproximation> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation vel;
      vel.f = velocityValues.segment<6>(6 * i);
      vel.dfdx = velocityJacobian.block(6 * i, 0, 6, state.rows());
      vel.dfdu = velocityJacobian.block(6 * i, state.rows(), 6, input.rows());
      velocities.emplace_back(std::move(vel));
    }
    return velocities;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getLinearAccelerationCppAd(const ocs2::ad_vector_t& state, const ocs2::ad_vector_t& input) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = mappingPtr_->getGeneralizedCoordinates(state);
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(state, input);
    const ocs2::ad_vector_t a = computeGeneralizedAccelerations<ocs2::ad_scalar_t>(state, input, pinocchioInterfaceCppAd_, *mappingPtr_);

    pinocchio::forwardKinematics(model, data, q, v, a);

    ocs2::ad_vector_t accelerations(3 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      accelerations.segment<3>(3 * i) = pinocchio::getFrameClassicalAcceleration(model, data, frameId, rf).linear();
    }
    return accelerations;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getLinearAcceleration(const vector_t& state,
                                                                const vector_t& input) const -> std::vector<vector3_t> {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t accelerationValues = linearAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput);

    std::vector<vector3_t> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      accelerations.emplace_back(accelerationValues.segment<3>(3 * i));
    }
    return accelerations;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getLinearAccelerationLinearApproximation(
                                                                                                                                   const vector_t& state, const vector_t& input) const {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t accelerationValues = linearAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput);
    const ocs2::matrix_t accelerationJacobian = linearAccelerationCppAdInterfacePtr_->getJacobian(stateInput);

    std::vector<ocs2::VectorFunctionLinearApproximation> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation acc;
      acc.f = accelerationValues.segment<3>(3 * i);
      acc.dfdx = accelerationJacobian.block(3 * i, 0, 3, state.rows());
      acc.dfdu = accelerationJacobian.block(3 * i, state.rows(), 3, input.rows());
      accelerations.emplace_back(std::move(acc));
    }
    return accelerations;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getAngularAccelerationCppAd(const ocs2::ad_vector_t& state, const ocs2::ad_vector_t& input) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = mappingPtr_->getGeneralizedCoordinates(state);
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(state, input);
    const ocs2::ad_vector_t a = computeGeneralizedAccelerations<ocs2::ad_scalar_t>(state, input, pinocchioInterfaceCppAd_, *mappingPtr_);

    pinocchio::forwardKinematics(model, data, q, v, a);

    ocs2::ad_vector_t accelerations(3 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      accelerations.segment<3>(3 * i) = pinocchio::getFrameClassicalAcceleration(model, data, frameId, rf).angular();
    }
    return accelerations;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getAngularAcceleration(const vector_t& state,
                                                                 const vector_t& input) const -> std::vector<vector3_t> {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t accelerationValues = angularAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput);

    std::vector<vector3_t> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      accelerations.emplace_back(accelerationValues.segment<3>(3 * i));
    }
    return accelerations;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getAngularAccelerationLinearApproximation(
                                                                                                                                    const vector_t& state, const vector_t& input) const {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t accelerationValues = angularAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput);
    const ocs2::matrix_t accelerationJacobian = angularAccelerationCppAdInterfacePtr_->getJacobian(stateInput);

    std::vector<ocs2::VectorFunctionLinearApproximation> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation acc;
      acc.f = accelerationValues.segment<3>(3 * i);
      acc.dfdx = accelerationJacobian.block(3 * i, 0, 3, state.rows());
      acc.dfdu = accelerationJacobian.block(3 * i, state.rows(), 3, input.rows());
      accelerations.emplace_back(std::move(acc));
    }
    return accelerations;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getAccelerationsCppAd(const ocs2::ad_vector_t& state, const ocs2::ad_vector_t& input) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = mappingPtr_->getGeneralizedCoordinates(state);
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(state, input);
    const ocs2::ad_vector_t a = computeGeneralizedAccelerations<ocs2::ad_scalar_t>(state, input, pinocchioInterfaceCppAd_, *mappingPtr_);

    pinocchio::forwardKinematics(model, data, q, v, a);

    ocs2::ad_vector_t accelerations(6 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      auto motion = pinocchio::getFrameClassicalAcceleration(model, data, frameId, rf);
      ocs2::ad_vector_t currAcceleration(6);
      currAcceleration.head(3) = motion.linear();
      currAcceleration.tail(3) = motion.angular();
      accelerations.segment<6>(6 * i) = currAcceleration;
    }
    return accelerations;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getAccelerations(const vector_t& state, const vector_t& input) const -> std::vector<vector6_t> {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t velocityValues = accelerationsCppAdInterfacePtr_->getFunctionValue(stateInput);

    std::vector<vector6_t> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      accelerations.emplace_back(velocityValues.segment<6>(6 * i));
    }
    return accelerations;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getAccelerationsLinearApproximation(
                                                                                                                              const vector_t& state, const vector_t& input) const {
    vector_t stateInput(state.rows() + input.rows());
    stateInput << state, input;
    const vector_t velocityValues = accelerationsCppAdInterfacePtr_->getFunctionValue(stateInput);
    const ocs2::matrix_t velocityJacobian = accelerationsCppAdInterfacePtr_->getJacobian(stateInput);

    std::vector<ocs2::VectorFunctionLinearApproximation> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation acc;
      acc.f = velocityValues.segment<6>(6 * i);
      acc.dfdx = velocityJacobian.block(6 * i, 0, 6, state.rows());
      acc.dfdu = velocityJacobian.block(6 * i, state.rows(), 6, input.rows());
      accelerations.emplace_back(std::move(acc));
    }
    return accelerations;
  }

}
