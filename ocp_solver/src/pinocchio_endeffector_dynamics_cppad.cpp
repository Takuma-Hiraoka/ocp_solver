#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "ocp_solver/pinocchio_endeffector_dynamics_cppad.h"
#include "ocp_solver/dynamics_helper_functions.h"

#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace ocp_solver {

  PinocchioEndEffectorDynamicsCppAd::PinocchioEndEffectorDynamicsCppAd(const ocs2::PinocchioInterface& pinocchioInterface,
                                                                       StateConverter<ocs2::ad_scalar_t>& stateConverter,
                                                                       std::vector<std::string> endEffectorIds,
                                                                       const std::string& modelName,
                                                                       const std::string& modelFolder,
                                                                       bool recompileLibraries,
                                                                       bool verbose)

  : endEffectorIds_(std::move(endEffectorIds)), pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()), mappingPtr_(&stateConverter) {
    for (const auto& bodyName : endEffectorIds_) {
      endEffectorFrameIds_.push_back(pinocchioInterface.getModel().getFrameId(bodyName));
    }

    size_t stateDim = mappingPtr_->getStateDim();
    size_t stateVariableDim = mappingPtr_->getStateVariableDim();
    size_t inputDim = mappingPtr_->getInputDim();

    // position function
    auto positionFunc = [&, this](const ocs2::ad_vector_t& x, const ocs2::ad_vector_t& p, ocs2::ad_vector_t& y) {
                          y = getPositionCppAd(x, p);
                        };
    positionCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(positionFunc, stateVariableDim, stateDim, modelName + "_position", modelFolder));

    // velocity function
    auto velocityFunc = [&, this](const ocs2::ad_vector_t& x, const ocs2::ad_vector_t& p, ocs2::ad_vector_t& y) {
                          const ocs2::ad_vector_t state = x.head(stateVariableDim);
                          const ocs2::ad_vector_t input = x.tail(inputDim);
                          y = getVelocityCppAd(state, input, p);
                        };
    velocityCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(velocityFunc, stateVariableDim + inputDim, stateDim, modelName + "_velocity", modelFolder));

    // orientation function
    auto orientationFunc = [&, this](const ocs2::ad_vector_t& x, const ocs2::ad_vector_t& p, ocs2::ad_vector_t& y) {
                             y = getOrientationCppAd(x, p);
                           };
    orientationCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(orientationFunc, stateVariableDim, stateDim, modelName + "_orientation", modelFolder));

    // orientation function
    auto orientationErrorFunc = [&, this](const ocs2::ad_vector_t& x, const ocs2::ad_vector_t& params, ocs2::ad_vector_t& y) {
                                  y = getOrientationErrorCppAd(x, params);
                                };
    orientationErrorCppAdInterfacePtr_.reset(
                                             new ocs2::CppAdInterface(orientationErrorFunc, stateVariableDim, stateDim + 4 * endEffectorFrameIds_.size(), modelName + "_orientationError", modelFolder));

    // velocity function
    auto angularVelocityFunc = [&, this](const ocs2::ad_vector_t& x, const ocs2::ad_vector_t& p, ocs2::ad_vector_t& y) {
                                 const ocs2::ad_vector_t state = x.head(stateVariableDim);
                                 const ocs2::ad_vector_t input = x.tail(inputDim);
                                 y = getAngularVelocityCppAd(state, input, p);
                               };
    angularVelocityCppAdInterfacePtr_.reset(
                                            new ocs2::CppAdInterface(angularVelocityFunc, stateVariableDim + inputDim, stateDim, modelName + "_angular_velocity", modelFolder));

    // twist function
    auto twistFunc = [&, this](const ocs2::ad_vector_t& x, const ocs2::ad_vector_t& p, ocs2::ad_vector_t& y) {
                       const ocs2::ad_vector_t state = x.head(stateVariableDim);
                       const ocs2::ad_vector_t input = x.tail(inputDim);
                       y = getTwistCppAd(state, input, p);
                     };
    twistCppAdInterfacePtr_.reset(new ocs2::CppAdInterface(twistFunc, stateVariableDim + inputDim, stateDim, modelName + "_twist", modelFolder));

    // linear acceleration function
    auto linearAccelerationFunc = [&, this](const ocs2::ad_vector_t& x, const ocs2::ad_vector_t& p, ocs2::ad_vector_t& y) {
                                    const ocs2::ad_vector_t state = x.head(stateVariableDim);
                                    const ocs2::ad_vector_t input = x.tail(inputDim);
                                    y = getLinearAccelerationCppAd(state, input, p);
                                  };
    linearAccelerationCppAdInterfacePtr_.reset(
                                               new ocs2::CppAdInterface(linearAccelerationFunc, stateVariableDim + inputDim, stateDim, modelName + "_linear_acceleration", modelFolder));

    // velocity function
    auto angularAccelerationFunc = [&, this](const ocs2::ad_vector_t& x, const ocs2::ad_vector_t& p, ocs2::ad_vector_t& y) {
                                     const ocs2::ad_vector_t state = x.head(stateVariableDim);
                                     const ocs2::ad_vector_t input = x.tail(inputDim);
                                     y = getAngularAccelerationCppAd(state, input, p);
                                   };
    angularAccelerationCppAdInterfacePtr_.reset(
                                                new ocs2::CppAdInterface(angularAccelerationFunc, stateVariableDim + inputDim, stateDim, modelName + "_angular_acceleration", modelFolder));

    // twist function
    auto accelerationsFunc = [&, this](const ocs2::ad_vector_t& x, const ocs2::ad_vector_t& p, ocs2::ad_vector_t& y) {
                               const ocs2::ad_vector_t state = x.head(stateVariableDim);
                               const ocs2::ad_vector_t input = x.tail(inputDim);
                               y = getAccelerationsCppAd(state, input, p);
                             };
    accelerationsCppAdInterfacePtr_.reset(
                                          new ocs2::CppAdInterface(accelerationsFunc, stateVariableDim + inputDim, stateDim, modelName + "_accelerations", modelFolder));

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

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getPositionCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& p) {
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = pinocchio::integrate(model, mappingPtr_->getGeneralizedCoordinates(p), dx.head(model.nv));

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
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    const vector_t positionValues = positionCppAdInterfacePtr_->getFunctionValue(dx, state);

    std::vector<vector3_t> positions;
    positions.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      positions.emplace_back(positionValues.segment<3>(3 * i));
    }
    return positions;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getPositionLinearApproximation(
                                                                                                                         const vector_t& state) const {
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    const vector_t positionValues = positionCppAdInterfacePtr_->getFunctionValue(dx, state);
    const ocs2::matrix_t positionJacobian = positionCppAdInterfacePtr_->getJacobian(dx, state);

    std::vector<ocs2::VectorFunctionLinearApproximation> positions;
    positions.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation pos;
      pos.f = positionValues.segment<3>(3 * i);
      pos.dfdx = positionJacobian.block(3 * i, 0, 3, dx.rows());
      positions.emplace_back(std::move(pos));
    }
    return positions;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getVelocityCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = pinocchio::integrate(model, mappingPtr_->getGeneralizedCoordinates(p), dx.head(model.nv));
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(p, input) + dx.tail(model.nv);

    pinocchio::forwardKinematics(model, data, q, v);

    ocs2::ad_vector_t velocities(3 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      velocities.segment<3>(3 * i) = pinocchio::getFrameVelocity(model, data, frameId, rf).linear();
    }
    return velocities;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getVelocity(const vector_t& state, const vector_t& input) const -> std::vector<vector3_t> {
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t velocityValues = velocityCppAdInterfacePtr_->getFunctionValue(stateInput, state);

    std::vector<vector3_t> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      velocities.emplace_back(velocityValues.segment<3>(3 * i));
    }
    return velocities;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getVelocityLinearApproximation(
                                                                                                                         const vector_t& state, const vector_t& input) const {
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t velocityValues = velocityCppAdInterfacePtr_->getFunctionValue(stateInput, state);
    const ocs2::matrix_t velocityJacobian = velocityCppAdInterfacePtr_->getJacobian(stateInput, state);

    std::vector<ocs2::VectorFunctionLinearApproximation> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation vel;
      vel.f = velocityValues.segment<3>(3 * i);
      vel.dfdx = velocityJacobian.block(3 * i, 0, 3, dx.rows());
      vel.dfdu = velocityJacobian.block(3 * i, dx.rows(), 3, input.rows());
      velocities.emplace_back(std::move(vel));
    }
    return velocities;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getOrientation(const vector_t& state) const -> std::vector<quaternion_t> {
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    const vector_t orientationValues = orientationCppAdInterfacePtr_->getFunctionValue(dx, state);

    std::vector<quaternion_t> orientations;
    orientations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      orientations.emplace_back(quaternion_t(orientationValues.segment<4>(4 * i)));
    }
    return orientations;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getOrientationCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& p) {
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = pinocchio::integrate(model, mappingPtr_->getGeneralizedCoordinates(p), dx.head(model.nv));

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
    vector_t params(state.rows() + 4 * endEffectorIds_.size());
    params.head(state.rows()) = state;
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      params.segment<4>(state.rows() + i * 4) = referenceOrientations[i].coeffs();
    }

    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    const vector_t errorValues = orientationErrorCppAdInterfacePtr_->getFunctionValue(dx, params);

    std::vector<vector3_t> errors;
    errors.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      errors.emplace_back(errorValues.segment<3>(3 * i));
    }
    return errors;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getOrientationErrorLinearApproximation(
                                                                                                                                 const vector_t& state, const std::vector<quaternion_t>& referenceOrientations) const {
    vector_t params(state.rows() + 4 * endEffectorIds_.size());
    params.head(state.rows()) = state;
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      params.segment<4>(state.rows() + i * 4) = referenceOrientations[i].coeffs();
    }

    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    const vector_t errorValues = orientationErrorCppAdInterfacePtr_->getFunctionValue(dx, params);
    const ocs2::matrix_t errorJacobian = orientationErrorCppAdInterfacePtr_->getJacobian(dx, params);

    std::vector<ocs2::VectorFunctionLinearApproximation> errors;
    errors.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation err;
      err.f = errorValues.segment<3>(3 * i);
      err.dfdx = errorJacobian.block(3 * i, 0, 3, dx.rows());
      errors.emplace_back(std::move(err));
    }
    return errors;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getOrientationErrorCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& params) {
    using ad_quaternion_t = Eigen::Quaternion<ocs2::ad_scalar_t>;

    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = pinocchio::integrate(model, mappingPtr_->getGeneralizedCoordinates(params.head(model.nq)), dx.head(model.nv));

    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);

    ocs2::ad_vector_t errors(3 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      // TODO
      const auto& R = data.oMf[frameId].rotation();

      ad_quaternion_t eeReferenceOrientation;
      eeReferenceOrientation.coeffs() = params.segment<4>(model.nq + 4 * i);
      Eigen::Matrix<ocs2::ad_scalar_t,3,3> R_ref =
        eeReferenceOrientation.toRotationMatrix();

      Eigen::Matrix<ocs2::ad_scalar_t,3,3> R_err =
        R_ref.transpose() * R;

      Eigen::Matrix<ocs2::ad_scalar_t,3,1> rotError;

      rotError <<
        R_err(2,1) - R_err(1,2),
        R_err(0,2) - R_err(2,0),
        R_err(1,0) - R_err(0,1);

      rotError *= ocs2::ad_scalar_t(0.5);

      errors.segment<3>(3*i) = rotError;
      // const ad_quaternion_t eeOrientation = ocs2::matrixToQuaternion(data.oMf[frameId].rotation());
      // ad_quaternion_t eeReferenceOrientation;
      // eeReferenceOrientation.coeffs() = params.segment<4>(model.nq + 4 * i);
      // errors.segment<3>(3 * i) = ocs2::quaternionDistance(eeOrientation, eeReferenceOrientation);
    }
    return errors;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getAngularVelocityCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = pinocchio::integrate(model, mappingPtr_->getGeneralizedCoordinates(p), dx.head(model.nv));
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(p, input) + dx.tail(model.nv);

    pinocchio::forwardKinematics(model, data, q, v);

    ocs2::ad_vector_t angularVelocities(3 * endEffectorFrameIds_.size());
    for (size_t i = 0; i < endEffectorFrameIds_.size(); i++) {
      const size_t frameId = endEffectorFrameIds_[i];
      angularVelocities.segment<3>(3 * i) = pinocchio::getFrameVelocity(model, data, frameId, rf).angular();
    }
    return angularVelocities;
  }

  auto PinocchioEndEffectorDynamicsCppAd::getAngularVelocity(const vector_t& state, const vector_t& input) const -> std::vector<vector3_t> {
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t velocityValues = angularVelocityCppAdInterfacePtr_->getFunctionValue(stateInput, state);

    std::vector<vector3_t> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      velocities.emplace_back(velocityValues.segment<3>(3 * i));
    }
    return velocities;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getAngularVelocityLinearApproximation(
                                                                                                                                const vector_t& state, const vector_t& input) const {
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t velocityValues = angularVelocityCppAdInterfacePtr_->getFunctionValue(stateInput, state);
    const ocs2::matrix_t velocityJacobian = angularVelocityCppAdInterfacePtr_->getJacobian(stateInput, state);

    std::vector<ocs2::VectorFunctionLinearApproximation> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation vel;
      vel.f = velocityValues.segment<3>(3 * i);
      vel.dfdx = velocityJacobian.block(3 * i, 0, 3, dx.rows());
      vel.dfdu = velocityJacobian.block(3 * i, dx.rows(), 3, input.rows());
      velocities.emplace_back(std::move(vel));
    }
    return velocities;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getTwistCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = pinocchio::integrate(model, mappingPtr_->getGeneralizedCoordinates(p), dx.head(model.nv));
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(p, input) + dx.tail(model.nv);

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
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t velocityValues = twistCppAdInterfacePtr_->getFunctionValue(stateInput, state);

    std::vector<vector6_t> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      velocities.emplace_back(velocityValues.segment<6>(6 * i));
    }
    return velocities;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getTwistLinearApproximation(const vector_t& state,
                                                                                                                      const vector_t& input) const {
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t velocityValues = twistCppAdInterfacePtr_->getFunctionValue(stateInput, state);
    const ocs2::matrix_t velocityJacobian = twistCppAdInterfacePtr_->getJacobian(stateInput, state);

    std::vector<ocs2::VectorFunctionLinearApproximation> velocities;
    velocities.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation vel;
      vel.f = velocityValues.segment<6>(6 * i);
      vel.dfdx = velocityJacobian.block(6 * i, 0, 6, dx.rows());
      vel.dfdu = velocityJacobian.block(6 * i, dx.rows(), 6, input.rows());
      velocities.emplace_back(std::move(vel));
    }
    return velocities;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getLinearAccelerationCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = pinocchio::integrate(model, mappingPtr_->getGeneralizedCoordinates(p), dx.head(model.nv));
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(p, input) + dx.tail(model.nv);
    const ocs2::ad_vector_t a = computeGeneralizedAccelerations<ocs2::ad_scalar_t>(p, input, pinocchioInterfaceCppAd_, *mappingPtr_);

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
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t accelerationValues = linearAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput, state);

    std::vector<vector3_t> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      accelerations.emplace_back(accelerationValues.segment<3>(3 * i));
    }
    return accelerations;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getLinearAccelerationLinearApproximation(
                                                                                                                                   const vector_t& state, const vector_t& input) const {
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t accelerationValues = linearAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput, state);
    const ocs2::matrix_t accelerationJacobian = linearAccelerationCppAdInterfacePtr_->getJacobian(stateInput, state);

    std::vector<ocs2::VectorFunctionLinearApproximation> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation acc;
      acc.f = accelerationValues.segment<3>(3 * i);
      acc.dfdx = accelerationJacobian.block(3 * i, 0, 3, dx.rows());
      acc.dfdu = accelerationJacobian.block(3 * i, dx.rows(), 3, input.rows());
      accelerations.emplace_back(std::move(acc));
    }
    return accelerations;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getAngularAccelerationCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = pinocchio::integrate(model, mappingPtr_->getGeneralizedCoordinates(p), dx.head(model.nv));
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(p, input) + dx.tail(model.nv);
    const ocs2::ad_vector_t a = computeGeneralizedAccelerations<ocs2::ad_scalar_t>(p, input, pinocchioInterfaceCppAd_, *mappingPtr_);

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
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t accelerationValues = angularAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput, state);

    std::vector<vector3_t> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      accelerations.emplace_back(accelerationValues.segment<3>(3 * i));
    }
    return accelerations;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getAngularAccelerationLinearApproximation(
                                                                                                                                    const vector_t& state, const vector_t& input) const {
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t accelerationValues = angularAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput, state);
    const ocs2::matrix_t accelerationJacobian = angularAccelerationCppAdInterfacePtr_->getJacobian(stateInput, state);

    std::vector<ocs2::VectorFunctionLinearApproximation> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation acc;
      acc.f = accelerationValues.segment<3>(3 * i);
      acc.dfdx = accelerationJacobian.block(3 * i, 0, 3, dx.rows());
      acc.dfdu = accelerationJacobian.block(3 * i, dx.rows(), 3, input.rows());
      accelerations.emplace_back(std::move(acc));
    }
    return accelerations;
  }

  ocs2::ad_vector_t PinocchioEndEffectorDynamicsCppAd::getAccelerationsCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p) {
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const auto& model = pinocchioInterfaceCppAd_.getModel();
    auto& data = pinocchioInterfaceCppAd_.getData();
    const ocs2::ad_vector_t q = pinocchio::integrate(model, mappingPtr_->getGeneralizedCoordinates(p), dx.head(model.nv));
    const ocs2::ad_vector_t v = mappingPtr_->getGeneralizedVelocities(p, input) + dx.tail(model.nv);
    const ocs2::ad_vector_t a = computeGeneralizedAccelerations<ocs2::ad_scalar_t>(p, input, pinocchioInterfaceCppAd_, *mappingPtr_);

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
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t velocityValues = accelerationsCppAdInterfacePtr_->getFunctionValue(stateInput, state);

    std::vector<vector6_t> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      accelerations.emplace_back(velocityValues.segment<6>(6 * i));
    }
    return accelerations;
  }

  std::vector<ocs2::VectorFunctionLinearApproximation> PinocchioEndEffectorDynamicsCppAd::getAccelerationsLinearApproximation(
                                                                                                                              const vector_t& state, const vector_t& input) const {
    const vector_t dx = vector_t::Zero(2*pinocchioInterfaceCppAd_.getModel().nv);
    vector_t stateInput(dx.rows() + input.rows());
    stateInput << dx, input;
    const vector_t velocityValues = accelerationsCppAdInterfacePtr_->getFunctionValue(stateInput, state);
    const ocs2::matrix_t velocityJacobian = accelerationsCppAdInterfacePtr_->getJacobian(stateInput, state);

    std::vector<ocs2::VectorFunctionLinearApproximation> accelerations;
    accelerations.reserve(endEffectorIds_.size());
    for (size_t i = 0; i < endEffectorIds_.size(); i++) {
      ocs2::VectorFunctionLinearApproximation acc;
      acc.f = velocityValues.segment<6>(6 * i);
      acc.dfdx = velocityJacobian.block(6 * i, 0, 6, dx.rows());
      acc.dfdu = velocityJacobian.block(6 * i, dx.rows(), 6, input.rows());
      accelerations.emplace_back(std::move(acc));
    }
    return accelerations;
  }

}
