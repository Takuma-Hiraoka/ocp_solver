#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "ocp_solver/pinocchio/pinocchio_frame_dynamics.h"
#include "ocp_solver/solver/dynamics_helper_functions.h"

#include <ocs2_robotic_tools/common/AngularVelocityMapping.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

namespace ocp_solver {

  namespace {
    struct FrameClassicalAccelerationDerivatives {
      ocs2::matrix_t dfdq;
      ocs2::matrix_t dfdv;
      ocs2::matrix_t dfda;
    };

    FrameClassicalAccelerationDerivatives computeFrameClassicalAccelerationDerivatives(
      const pinocchio::Model& model,
      pinocchio::Data& data,
      pinocchio::JointIndex parentJointIndex,
      const pinocchio::SE3& localPose,
      pinocchio::ReferenceFrame referenceFrame) {
      ocs2::matrix_t v_partial_dq(6, model.nv);
      ocs2::matrix_t spatial_dq(6, model.nv);
      ocs2::matrix_t spatial_dv(6, model.nv);
      ocs2::matrix_t spatial_da(6, model.nv);
      pinocchio::getFrameAccelerationDerivatives(model, data, parentJointIndex, localPose, referenceFrame,
                                                 v_partial_dq, spatial_dq, spatial_dv, spatial_da);

      FrameClassicalAccelerationDerivatives derivatives{spatial_dq, spatial_dv, spatial_da};
      if (!derivatives.dfda.allFinite()) {
        derivatives.dfda = derivatives.dfda.unaryExpr([](double x) {
          return std::isfinite(x) ? x : 0.0;
        });
      }

      const pinocchio::Motion frameVelocity = pinocchio::getFrameVelocity(model, data, parentJointIndex, localPose, referenceFrame);
      const Eigen::Vector3d omega = frameVelocity.angular();
      const Eigen::Vector3d linearVelocity = frameVelocity.linear();
      const ocs2::matrix_t S_omega = ocs2::skewSymmetricMatrix(omega);
      const ocs2::matrix_t S_linearVelocity = ocs2::skewSymmetricMatrix(linearVelocity);

      for (int k = 0; k < model.nv; ++k) {
        derivatives.dfdq.block<3,1>(0, k) +=
          -S_linearVelocity * spatial_dq.block<3,1>(3, k) + S_omega * spatial_dq.block<3,1>(0, k);
        derivatives.dfdv.block<3,1>(0, k) +=
          -S_linearVelocity * spatial_dv.block<3,1>(3, k) + S_omega * spatial_dv.block<3,1>(0, k);
      }

      return derivatives;
    }

    FrameClassicalAccelerationDerivatives computeFrameClassicalAccelerationDerivatives(
      const pinocchio::Model& model,
      pinocchio::Data& data,
      pinocchio::FrameIndex frameId,
      pinocchio::ReferenceFrame referenceFrame) {
      const pinocchio::Frame& frame = model.frames[frameId];
      return computeFrameClassicalAccelerationDerivatives(model, data, frame.parentJoint, frame.placement, referenceFrame);
    }

    void addBaseAccelerationChainRule(const OCPPreComputation& preComputation,
                                      const StateConverter<ocs2::scalar_t>& stateConverter,
                                      const ocs2::matrix_t& frameAccelerationPartialDa,
                                      size_t rowOffset,
                                      size_t rowCount,
                                      pinocchio::ReferenceFrame referenceFrame,
                                      ocs2::VectorFunctionLinearApproximation& acceleration) {
      const size_t baseVDim = stateConverter.getBaseVDim();
      if (baseVDim == 0) {
        return;
      }

      ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
      const BaseAccelerationLinearApproximation baseAccelerationApprox =
        (referenceFrame == pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED)
          ? preComputation.getBaseAccelerationLinearApproximation()
          : computeBaseAccelerationLinearApproximation(preComputation.getGeneralizedCoordinates(),
                                                       preComputation.getGeneralizedVelocities(),
                                                       preComputation.getGeneralizedAccelerations(),
                                                       preComputation.getInput(),
                                                       pinocchioInterface,
                                                       stateConverter,
                                                       referenceFrame);
      const ocs2::matrix_t framePartialBaseAcceleration =
        frameAccelerationPartialDa.block(rowOffset, 0, rowCount, baseVDim);
      acceleration.dfdx.leftCols(stateConverter.getTangentDim()).noalias() +=
        framePartialBaseAcceleration * baseAccelerationApprox.dfdq;
      acceleration.dfdx.block(0, stateConverter.getTangentDim(), acceleration.dfdx.rows(), stateConverter.getTangentDim()).noalias() +=
        framePartialBaseAcceleration * baseAccelerationApprox.dfdv;
      acceleration.dfdu.noalias() += framePartialBaseAcceleration * baseAccelerationApprox.dfdu;
    }
  }

  PinocchioFrameDynamics::PinocchioFrameDynamics(const ocs2::PinocchioInterface& pinocchioInterface,
                                                 StateConverter<ocs2::scalar_t>& stateConverter,
                                                 std::string frameName)
    : stateConverter_(&stateConverter), frameName_(std::move(frameName)) {
    frameId_ = pinocchioInterface.getModel().getFrameId(frameName_);
  }

  PinocchioFrameDynamics::PinocchioFrameDynamics(StateConverter<ocs2::scalar_t>& stateConverter, size_t contactIndex)
    : frameName_(stateConverter.getContactCandidate(contactIndex).frameName),
      frameId_(stateConverter.getContactCandidate(contactIndex).index),
      contactIndex_(contactIndex),
      useContactCandidate_(true),
      stateConverter_(&stateConverter) {}

  PinocchioFrameDynamics::PinocchioFrameDynamics(const PinocchioFrameDynamics& rhs)
    : frameName_(rhs.frameName_),
      frameId_(rhs.frameId_),
      contactIndex_(rhs.contactIndex_),
      useContactCandidate_(rhs.useContactCandidate_),
      stateConverter_(rhs.stateConverter_) {}

  PinocchioFrameDynamics* PinocchioFrameDynamics::clone() const {
    return new PinocchioFrameDynamics(*this);
  }

  ocs2::vector_t PinocchioFrameDynamics::getPosition(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    if (useContactCandidate_) {
      return getContactCandidatePlacement(pinocchioInterface, stateConverter_->getContactCandidate(contactIndex_)).translation();
    }
    pinocchio::updateFramePlacement(model, data, frameId_);
    return pinocchioInterface.getData().oMf[frameId_].translation();
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getPositionLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    ocs2::VectorFunctionLinearApproximation position;
    if (useContactCandidate_) {
      position.f = getContactCandidatePlacement(pinocchioInterface, stateConverter_->getContactCandidate(contactIndex_)).translation();
    }
    else {
      pinocchio::updateFramePlacement(model, data, frameId_);
      position.f = data.oMf[frameId_].translation();
    }

    ocs2::matrix_t J = ocs2::matrix_t::Zero(6, model.nv);
    if (useContactCandidate_) {
      getContactCandidateJacobian(pinocchioInterface, stateConverter_->getContactCandidate(contactIndex_), rf, J);
    } else {
      pinocchio::getFrameJacobian(model, data, frameId_, rf, J);
    }
    position.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    position.dfdx.leftCols(stateConverter_->getTangentDim()) = J.topRows(3);
    return position;
  }

  ocs2::vector_t PinocchioFrameDynamics::getVelocity(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    if (useContactCandidate_) {
      return pinocchio::getFrameVelocity(model, data, stateConverter_->getContactCandidate(contactIndex_).parentJointIndex,
                                         stateConverter_->getContactCandidate(contactIndex_).localPose, rf).linear();
    }
    return pinocchio::getFrameVelocity(model, data, frameId_, rf).linear();
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getVelocityLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    if (useContactCandidate_) {
      const pinocchio::Motion frameVel =
        pinocchio::getFrameVelocity(model, data, stateConverter_->getContactCandidate(contactIndex_).parentJointIndex,
                                    stateConverter_->getContactCandidate(contactIndex_).localPose, rf);
      ocs2::matrix_t J = ocs2::matrix_t::Zero(6, model.nv);
      getContactCandidateJacobian(pinocchioInterface, stateConverter_->getContactCandidate(contactIndex_), rf, J);
      ocs2::VectorFunctionLinearApproximation vel;
      vel.f = frameVel.linear();
      vel.dfdx.setZero(3, stateConverter_->getStateVariableDim());
      vel.dfdx.block(0, stateConverter_->getTangentDim(), 3, stateConverter_->getTangentDim()) = J.topRows(3);
      return vel;
    }

    ocs2::matrix_t v_partial_dq = ocs2::matrix_t::Zero(6, model.nv);
    ocs2::matrix_t v_partial_dv = ocs2::matrix_t::Zero(6, model.nv);
    pinocchio::getFrameVelocityDerivatives(model, data, frameId_, rf, v_partial_dq, v_partial_dv);
    const pinocchio::Motion frameVel = pinocchio::getFrameVelocity(model, data, frameId_, rf);
    // For reference frame LOCAL_WORLD_ALIGNED the jacobian needs to be corrected.
    if (rf == pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED) {
      v_partial_dq.topRows<3>() += ocs2::skewSymmetricMatrix(vector3_t(frameVel.angular())) * v_partial_dv.topRows<3>();
    }
    ocs2::VectorFunctionLinearApproximation vel;
    vel.f = frameVel.linear();
    vel.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    vel.dfdx.leftCols(stateConverter_->getTangentDim()) = v_partial_dq.topRows(3);
    vel.dfdx.block(0, stateConverter_->getTangentDim(), 3, stateConverter_->getTangentDim()) = v_partial_dv.topRows(3);

    return vel;
  }

  PinocchioFrameDynamics::quaternion_t PinocchioFrameDynamics::getOrientation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    if (useContactCandidate_) {
      return ocs2::matrixToQuaternion(
        getContactCandidatePlacement(pinocchioInterface, stateConverter_->getContactCandidate(contactIndex_)).rotation());
    }
    pinocchio::updateFramePlacement(model, data, frameId_);
    return ocs2::matrixToQuaternion(pinocchioInterface.getData().oMf[frameId_].rotation());
  }

  ocs2::vector_t PinocchioFrameDynamics::getOrientationError(const OCPPreComputation& preComputation, const quaternion_t& referenceOrientation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    if (useContactCandidate_) {
      return pinocchio::log3(referenceOrientation.toRotationMatrix().transpose() *
                             getContactCandidatePlacement(pinocchioInterface, stateConverter_->getContactCandidate(contactIndex_)).rotation());
    }
    pinocchio::updateFramePlacement(model, data, frameId_);
    return pinocchio::log3(referenceOrientation.toRotationMatrix().transpose() * pinocchioInterface.getData().oMf[frameId_].rotation());
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getOrientationErrorLinearApproximation(const OCPPreComputation& preComputation, const quaternion_t& referenceOrientation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    ocs2::VectorFunctionLinearApproximation error;
    const Eigen::Matrix3d frameRotation = useContactCandidate_
      ? getContactCandidatePlacement(pinocchioInterface, stateConverter_->getContactCandidate(contactIndex_)).rotation()
      : [&]() {
          pinocchio::updateFramePlacement(model, data, frameId_);
          return data.oMf[frameId_].rotation();
        }();
    error.f = pinocchio::log3(referenceOrientation.toRotationMatrix().transpose() * frameRotation);
    ocs2::matrix_t J = ocs2::matrix_t::Zero(6, model.nv);
    if (useContactCandidate_) {
      getContactCandidateJacobian(pinocchioInterface, stateConverter_->getContactCandidate(contactIndex_), rf, J);
    } else {
      pinocchio::getFrameJacobian(model, data, frameId_, rf, J);
    }
    matrix3x_t Jlog;
    pinocchio::Jlog3(referenceOrientation.toRotationMatrix().transpose() * frameRotation, Jlog);
    error.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    error.dfdx.leftCols(stateConverter_->getTangentDim()) = Jlog * referenceOrientation.toRotationMatrix().transpose() * J.bottomRows<3>();
    return error;
  }

  ocs2::vector_t PinocchioFrameDynamics::getAngularVelocity(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    if (useContactCandidate_) {
      return pinocchio::getFrameVelocity(model, data, stateConverter_->getContactCandidate(contactIndex_).parentJointIndex,
                                         stateConverter_->getContactCandidate(contactIndex_).localPose, rf).angular();
    }
    return pinocchio::getFrameVelocity(model, data, frameId_, rf).angular();
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getAngularVelocityLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    ocs2::matrix_t v_partial_dq = ocs2::matrix_t::Zero(6, model.nv);
    ocs2::matrix_t v_partial_dv = ocs2::matrix_t::Zero(6, model.nv);
    pinocchio::Motion frameVel;
    if (useContactCandidate_) {
      const auto& contactCandidate = stateConverter_->getContactCandidate(contactIndex_);
      pinocchio::getFrameVelocityDerivatives(model, data, contactCandidate.parentJointIndex,
                                             contactCandidate.localPose, rf, v_partial_dq, v_partial_dv);
      frameVel = pinocchio::getFrameVelocity(model, data, contactCandidate.parentJointIndex,
                                             contactCandidate.localPose, rf);
    } else {
      pinocchio::getFrameVelocityDerivatives(model, data, frameId_, rf, v_partial_dq, v_partial_dv);
      frameVel = pinocchio::getFrameVelocity(model, data, frameId_, rf);
    }
    ocs2::VectorFunctionLinearApproximation vel;
    vel.f = frameVel.angular();
    vel.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    vel.dfdx.leftCols(stateConverter_->getTangentDim()) = v_partial_dq.bottomRows(3);
    vel.dfdx.block(0, stateConverter_->getTangentDim(), 3, stateConverter_->getTangentDim()) = v_partial_dv.bottomRows(3);

    return vel;
  }

  ocs2::vector_t PinocchioFrameDynamics::getTwist(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    pinocchio::Motion frameVelocity;
    if (useContactCandidate_) {
      const auto& contactCandidate = stateConverter_->getContactCandidate(contactIndex_);
      frameVelocity = pinocchio::getFrameVelocity(model, data, contactCandidate.parentJointIndex,
                                                  contactCandidate.localPose, rf);
    } else {
      frameVelocity = pinocchio::getFrameVelocity(model, data, frameId_, rf);
    }
    vector6_t twist;
    twist.head(3) = frameVelocity.linear();
    twist.tail(3) = frameVelocity.angular();
    return twist;
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getTwistLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    ocs2::matrix_t v_partial_dq = ocs2::matrix_t::Zero(6, model.nv);
    ocs2::matrix_t v_partial_dv = ocs2::matrix_t::Zero(6, model.nv);
    pinocchio::Motion frameVel;
    if (useContactCandidate_) {
      const auto& contactCandidate = stateConverter_->getContactCandidate(contactIndex_);
      pinocchio::getFrameVelocityDerivatives(model, data, contactCandidate.parentJointIndex,
                                             contactCandidate.localPose, rf, v_partial_dq, v_partial_dv);
      frameVel = pinocchio::getFrameVelocity(model, data, contactCandidate.parentJointIndex,
                                             contactCandidate.localPose, rf);
    } else {
      pinocchio::getFrameVelocityDerivatives(model, data, frameId_, rf, v_partial_dq, v_partial_dv);
      frameVel = pinocchio::getFrameVelocity(model, data, frameId_, rf);
    }
    // For reference frame LOCAL_WORLD_ALIGNED the jacobian needs to be corrected.
    if (rf == pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED) {
      v_partial_dq.topRows<3>() += ocs2::skewSymmetricMatrix(vector3_t(frameVel.angular())) * v_partial_dv.topRows<3>();
    }
    ocs2::VectorFunctionLinearApproximation twist;
    vector_t twist_f(6);
    twist_f.head(3) = frameVel.linear();
    twist_f.tail(3) = frameVel.angular();
    twist.f = twist_f;

    twist.dfdx.setZero(6, stateConverter_->getStateVariableDim());
    twist.dfdx.leftCols(stateConverter_->getTangentDim()) = v_partial_dq;
    twist.dfdx.block(0, stateConverter_->getTangentDim(), 6, stateConverter_->getTangentDim()) = v_partial_dv;
    return twist;
  }

  ocs2::vector_t PinocchioFrameDynamics::getLinearAcceleration(const OCPPreComputation& preComputation) const { 
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    if (useContactCandidate_) {
      const auto& contactCandidate = stateConverter_->getContactCandidate(contactIndex_);
      return pinocchio::getFrameClassicalAcceleration(model, data, contactCandidate.parentJointIndex,
                                                      contactCandidate.localPose, rf).linear();
    }
    return pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).linear();
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getLinearAccelerationLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    ocs2::VectorFunctionLinearApproximation acceleration;
    FrameClassicalAccelerationDerivatives derivatives;
    if (useContactCandidate_) {
      const auto& contactCandidate = stateConverter_->getContactCandidate(contactIndex_);
      acceleration.f = pinocchio::getFrameClassicalAcceleration(model, data, contactCandidate.parentJointIndex,
                                                                contactCandidate.localPose, rf).linear();
      derivatives = computeFrameClassicalAccelerationDerivatives(model, data, contactCandidate.parentJointIndex,
                                                                 contactCandidate.localPose, rf);
    } else {
      acceleration.f = pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).linear();
      derivatives = computeFrameClassicalAccelerationDerivatives(model, data, frameId_, rf);
    }

    acceleration.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    acceleration.dfdx.leftCols(stateConverter_->getTangentDim()) = derivatives.dfdq.topRows(3);
    acceleration.dfdx.block(0, stateConverter_->getTangentDim(), 3, stateConverter_->getTangentDim()) = derivatives.dfdv.topRows(3);

    acceleration.dfdu.setZero(3, stateConverter_->getInputDim());
    acceleration.dfdu.block(0, stateConverter_->getJointAccelerationsStartindex(), 3, stateConverter_->getJointDim()) =
      derivatives.dfda.block(0, stateConverter_->getBaseVDim(), 3, stateConverter_->getJointDim());
    addBaseAccelerationChainRule(preComputation, *stateConverter_, derivatives.dfda, 0, 3, rf, acceleration);
    return acceleration;
  }

  ocs2::vector_t PinocchioFrameDynamics::getAngularAcceleration(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    if (useContactCandidate_) {
      const auto& contactCandidate = stateConverter_->getContactCandidate(contactIndex_);
      return pinocchio::getFrameClassicalAcceleration(model, data, contactCandidate.parentJointIndex,
                                                      contactCandidate.localPose, rf).angular();
    }
    return pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).angular();
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getAngularAccelerationLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    ocs2::VectorFunctionLinearApproximation acceleration;
    FrameClassicalAccelerationDerivatives derivatives;
    if (useContactCandidate_) {
      const auto& contactCandidate = stateConverter_->getContactCandidate(contactIndex_);
      acceleration.f = pinocchio::getFrameClassicalAcceleration(model, data, contactCandidate.parentJointIndex,
                                                                contactCandidate.localPose, rf).angular();
      derivatives = computeFrameClassicalAccelerationDerivatives(model, data, contactCandidate.parentJointIndex,
                                                                 contactCandidate.localPose, rf);
    } else {
      acceleration.f = pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).angular();
      derivatives = computeFrameClassicalAccelerationDerivatives(model, data, frameId_, rf);
    }

    acceleration.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    acceleration.dfdx.leftCols(stateConverter_->getTangentDim()) = derivatives.dfdq.bottomRows(3);
    acceleration.dfdx.block(0, stateConverter_->getTangentDim(), 3, stateConverter_->getTangentDim()) = derivatives.dfdv.bottomRows(3);

    acceleration.dfdu.setZero(3, stateConverter_->getInputDim());
    acceleration.dfdu.block(0, stateConverter_->getJointAccelerationsStartindex(), 3, stateConverter_->getJointDim()) =
      derivatives.dfda.block(3, stateConverter_->getBaseVDim(), 3, stateConverter_->getJointDim());
    addBaseAccelerationChainRule(preComputation, *stateConverter_, derivatives.dfda, 3, 3, rf, acceleration);
    return acceleration;
  }


  ocs2::vector_t PinocchioFrameDynamics::getAccelerations(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    pinocchio::Motion frameAcceleration;
    if (useContactCandidate_) {
      const auto& contactCandidate = stateConverter_->getContactCandidate(contactIndex_);
      frameAcceleration = pinocchio::getFrameClassicalAcceleration(model, data, contactCandidate.parentJointIndex,
                                                                   contactCandidate.localPose, rf);
    } else {
      frameAcceleration = pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf);
    }
    vector6_t acceleration;
    acceleration.head(3) = frameAcceleration.linear();
    acceleration.tail(3) = frameAcceleration.angular();
    return acceleration;
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getAccelerationsLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    ocs2::VectorFunctionLinearApproximation acceleration;
    vector6_t acceleration_f;
    FrameClassicalAccelerationDerivatives derivatives;
    if (useContactCandidate_) {
      const auto& contactCandidate = stateConverter_->getContactCandidate(contactIndex_);
      const pinocchio::Motion frameAcceleration =
        pinocchio::getFrameClassicalAcceleration(model, data, contactCandidate.parentJointIndex,
                                                 contactCandidate.localPose, rf);
      acceleration_f.head(3) = frameAcceleration.linear();
      acceleration_f.tail(3) = frameAcceleration.angular();
      derivatives = computeFrameClassicalAccelerationDerivatives(model, data, contactCandidate.parentJointIndex,
                                                                 contactCandidate.localPose, rf);
    } else {
      const pinocchio::Motion frameAcceleration = pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf);
      acceleration_f.head(3) = frameAcceleration.linear();
      acceleration_f.tail(3) = frameAcceleration.angular();
      derivatives = computeFrameClassicalAccelerationDerivatives(model, data, frameId_, rf);
    }
    acceleration.f = acceleration_f;

    acceleration.dfdx.setZero(6, stateConverter_->getStateVariableDim());
    acceleration.dfdx.leftCols(stateConverter_->getTangentDim()) = derivatives.dfdq;
    acceleration.dfdx.block(0, stateConverter_->getTangentDim(), 6, stateConverter_->getTangentDim()) = derivatives.dfdv;
    acceleration.dfdu.setZero(6, stateConverter_->getInputDim());
    acceleration.dfdu.block(0, stateConverter_->getJointAccelerationsStartindex(), 6, stateConverter_->getJointDim()) =
      derivatives.dfda.block(0, stateConverter_->getBaseVDim(), 6, stateConverter_->getJointDim());
    addBaseAccelerationChainRule(preComputation, *stateConverter_, derivatives.dfda, 0, 6, rf, acceleration);

    return acceleration;
  }

}
