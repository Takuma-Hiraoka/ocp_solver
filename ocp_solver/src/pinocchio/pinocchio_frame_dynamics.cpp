#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "ocp_solver/pinocchio/pinocchio_frame_dynamics.h"
#include "ocp_solver/solver/dynamics_helper_functions.h"

#include <ocs2_robotic_tools/common/AngularVelocityMapping.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace ocp_solver {

  PinocchioFrameDynamics::PinocchioFrameDynamics(const ocs2::PinocchioInterface& pinocchioInterface,
                                                 StateConverter<ocs2::scalar_t>& stateConverter,
                                                 std::string frameName)
    : stateConverter_(&stateConverter), frameName_(std::move(frameName)) {
    frameId_ = pinocchioInterface.getModel().getFrameId(frameName_);
  }

  PinocchioFrameDynamics::PinocchioFrameDynamics(const PinocchioFrameDynamics& rhs)
    : frameName_(rhs.frameName_),
      frameId_(rhs.frameId_),
      stateConverter_(rhs.stateConverter_) {}

  PinocchioFrameDynamics* PinocchioFrameDynamics::clone() const {
    return new PinocchioFrameDynamics(*this);
  }

  ocs2::vector_t PinocchioFrameDynamics::getPosition(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    pinocchio::updateFramePlacement(model, data, frameId_);
    return pinocchioInterface.getData().oMf[frameId_].translation();
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getPositionLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    pinocchio::updateFramePlacement(model, data, frameId_);

    ocs2::VectorFunctionLinearApproximation position;
    position.f = data.oMf[frameId_].translation();

    ocs2::matrix_t J = ocs2::matrix_t::Zero(6, model.nv);
    pinocchio::getFrameJacobian(model, data, frameId_, rf, J);
    position.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    position.dfdx.leftCols(stateConverter_->getTangentDim()) = J.topRows(3);
    return position;
  }

  ocs2::vector_t PinocchioFrameDynamics::getVelocity(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    return pinocchio::getFrameVelocity(model, data, frameId_, rf).linear();
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getVelocityLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    ocs2::matrix_t v_partial_dq = ocs2::matrix_t::Zero(6, model.nv);
    ocs2::matrix_t v_partial_dv = ocs2::matrix_t::Zero(6, model.nv);
    pinocchio::getFrameVelocityDerivatives(model, data, frameId_, rf, v_partial_dq, v_partial_dv);
    const auto frameVel = pinocchio::getFrameVelocity(model, data, frameId_, rf);
    // For reference frame LOCAL_WORLD_ALIGNED the jacobian needs to be corrected.
    if (rf == pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED) {
      v_partial_dq.topRows<3>() += ocs2::skewSymmetricMatrix(vector3_t(frameVel.angular())) * v_partial_dv.topRows<3>();
    }
    ocs2::VectorFunctionLinearApproximation vel;
    vel.f = frameVel.linear();
    vel.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    vel.dfdx.leftCols(stateConverter_->getTangentDim()) = v_partial_dq.topRows(3);
    vel.dfdx.rightCols(stateConverter_->getTangentDim()) = v_partial_dv.topRows(3);

    return vel;
  }

  PinocchioFrameDynamics::quaternion_t PinocchioFrameDynamics::getOrientation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    pinocchio::updateFramePlacement(model, data, frameId_);
    return ocs2::matrixToQuaternion(pinocchioInterface.getData().oMf[frameId_].rotation());
  }

  ocs2::vector_t PinocchioFrameDynamics::getOrientationError(const OCPPreComputation& preComputation, const quaternion_t& referenceOrientation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    pinocchio::updateFramePlacement(model, data, frameId_);
    return pinocchio::log3(referenceOrientation.toRotationMatrix().transpose() * pinocchioInterface.getData().oMf[frameId_].rotation());
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getOrientationErrorLinearApproximation(const OCPPreComputation& preComputation, const quaternion_t& referenceOrientation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    pinocchio::updateFramePlacement(model, data, frameId_);

    ocs2::VectorFunctionLinearApproximation error;
    error.f = pinocchio::log3(referenceOrientation.toRotationMatrix().transpose() * data.oMf[frameId_].rotation());
    ocs2::matrix_t J = ocs2::matrix_t::Zero(6, model.nv);
    pinocchio::getFrameJacobian(model, data, frameId_, rf, J);
    matrix3x_t Jlog;
    pinocchio::Jlog3(referenceOrientation.toRotationMatrix().transpose() * data.oMf[frameId_].rotation(), Jlog);
    error.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    error.dfdx.leftCols(stateConverter_->getTangentDim()) = Jlog * J.bottomRows<3>();
    return error;
  }

  ocs2::vector_t PinocchioFrameDynamics::getAngularVelocity(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    return pinocchio::getFrameVelocity(model, data, frameId_, rf).angular();
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getAngularVelocityLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    ocs2::matrix_t v_partial_dq = ocs2::matrix_t::Zero(6, model.nv);
    ocs2::matrix_t v_partial_dv = ocs2::matrix_t::Zero(6, model.nv);
    pinocchio::getFrameVelocityDerivatives(model, data, frameId_, rf, v_partial_dq, v_partial_dv);
    const auto frameVel = pinocchio::getFrameVelocity(model, data, frameId_, rf);
    ocs2::VectorFunctionLinearApproximation vel;
    vel.f = frameVel.angular();
    vel.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    vel.dfdx.leftCols(stateConverter_->getTangentDim()) = v_partial_dq.bottomRows(3);
    vel.dfdx.rightCols(stateConverter_->getTangentDim()) = v_partial_dv.bottomRows(3);

    return vel;
  }

  ocs2::vector_t PinocchioFrameDynamics::getTwist(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    vector6_t twist;
    twist.head(3) = pinocchio::getFrameVelocity(model, data, frameId_, rf).linear();
    twist.tail(3) = pinocchio::getFrameVelocity(model, data, frameId_, rf).angular();
    return twist;
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getTwistLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    ocs2::matrix_t v_partial_dq = ocs2::matrix_t::Zero(6, model.nv);
    ocs2::matrix_t v_partial_dv = ocs2::matrix_t::Zero(6, model.nv);
    pinocchio::getFrameVelocityDerivatives(model, data, frameId_, rf, v_partial_dq, v_partial_dv);
    const auto frameVel = pinocchio::getFrameVelocity(model, data, frameId_, rf);
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
    twist.dfdx.rightCols(stateConverter_->getTangentDim()) = v_partial_dv;
    return twist;
  }

  ocs2::vector_t PinocchioFrameDynamics::getLinearAcceleration(const OCPPreComputation& preComputation) const { 
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    return pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).linear();
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getLinearAccelerationLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    ocs2::VectorFunctionLinearApproximation acceleration;
    acceleration.f = pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).linear();

    ocs2::matrix_t v_partial_dq(6, model.nv);
    ocs2::matrix_t spatial_dq(6, model.nv);
    ocs2::matrix_t spatial_dv(6, model.nv);
    ocs2::matrix_t spatial_da(6, model.nv);
    pinocchio::getFrameAccelerationDerivatives(model, data, frameId_, rf, v_partial_dq, spatial_dq, spatial_dv, spatial_da);

    ocs2::matrix_t a_partial_dq = spatial_dq;
    ocs2::matrix_t a_partial_dv = spatial_dv;
    ocs2::matrix_t a_partial_da = spatial_da;

    const pinocchio::Motion v_frame = pinocchio::getFrameVelocity(model, data, frameId_, rf);
    const vector3_t omega = v_frame.angular();
    const vector3_t vlin  = v_frame.linear();
    const ocs2::matrix_t S_omega = ocs2::skewSymmetricMatrix(omega);
    const ocs2::matrix_t S_vlin = ocs2::skewSymmetricMatrix(vlin);

    for (int k = 0; k < model.nv; ++k) {
      // wrt q
      const vector3_t dvlin_dq  = spatial_dq.block<3,1>(0, k);
      const vector3_t domega_dq = spatial_dq.block<3,1>(3, k);

      a_partial_dq.block<3,1>(0, k) +=
        -S_vlin * domega_dq + S_omega * dvlin_dq;

      // wrt v
      const vector3_t dvlin_dv  = spatial_dv.block<3,1>(0, k);
      const vector3_t domega_dv = spatial_dv.block<3,1>(3, k);

      a_partial_dv.block<3,1>(0, k) +=
        -S_vlin * domega_dv + S_omega * dvlin_dv;

      // wrt a:
      // no correction because omega and vlin do not depend on a.
    }

    acceleration.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    acceleration.dfdx.leftCols(stateConverter_->getTangentDim()) = a_partial_dq.topRows(3);
    acceleration.dfdx.rightCols(stateConverter_->getTangentDim()) = a_partial_dv.topRows(3);

    acceleration.dfdu.setZero(3, stateConverter_->getInputDim());
    if (stateConverter_->getBaseVDim() > 0) {
      for (int i=0; i<stateConverter_->contactCandidateIds.size(); i++) {
        ocs2::matrix_t J = ocs2::matrix_t::Zero(6, stateConverter_->getTangentDim());
        pinocchio::getFrameJacobian(model, data, stateConverter_->contactCandidateIds[i], rf, J);
        acceleration.dfdu.block(0,i*6, 3, 6) = (data.M.topLeftCorner(6,6).inverse() * J.block(0,0,6,6).transpose()).topRows(3);
      }
    }
    acceleration.dfdu.rightCols(stateConverter_->getJointDim()) = (a_partial_da.rightCols(stateConverter_->getJointDim()) + a_partial_da.leftCols(stateConverter_->getBaseVDim()) * data.M.topLeftCorner(6,6).inverse() * data.M.block(0, 6, 6, stateConverter_->getJointDim())).topRows(3);
    return acceleration;
  }

  ocs2::vector_t PinocchioFrameDynamics::getAngularAcceleration(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    return pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).angular();
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getAngularAccelerationLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    ocs2::VectorFunctionLinearApproximation acceleration;
    acceleration.f = pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).angular();

    ocs2::matrix_t v_partial_dq(6, model.nv);
    ocs2::matrix_t spatial_dq(6, model.nv);
    ocs2::matrix_t spatial_dv(6, model.nv);
    ocs2::matrix_t spatial_da(6, model.nv);
    pinocchio::getFrameAccelerationDerivatives(model, data, frameId_, rf, v_partial_dq, spatial_dq, spatial_dv, spatial_da);

    ocs2::matrix_t a_partial_dq = spatial_dq;
    ocs2::matrix_t a_partial_dv = spatial_dv;
    ocs2::matrix_t a_partial_da = spatial_da;
    if (!a_partial_da.allFinite()) {
      a_partial_da = a_partial_da.unaryExpr([](double x) {
                                              return std::isfinite(x) ? x : 0.0;
                                            });
    }

    const pinocchio::Motion v_frame = pinocchio::getFrameVelocity(model, data, frameId_, rf);
    const vector3_t omega = v_frame.angular();
    const vector3_t vlin  = v_frame.linear();
    const ocs2::matrix_t S_omega = ocs2::skewSymmetricMatrix(omega);
    const ocs2::matrix_t S_vlin = ocs2::skewSymmetricMatrix(vlin);

    for (int k = 0; k < model.nv; ++k) {
      // wrt q
      const vector3_t dvlin_dq  = spatial_dq.block<3,1>(0, k);
      const vector3_t domega_dq = spatial_dq.block<3,1>(3, k);

      a_partial_dq.block<3,1>(0, k) +=
        -S_vlin * domega_dq + S_omega * dvlin_dq;

      // wrt v
      const vector3_t dvlin_dv  = spatial_dv.block<3,1>(0, k);
      const vector3_t domega_dv = spatial_dv.block<3,1>(3, k);

      a_partial_dv.block<3,1>(0, k) +=
        -S_vlin * domega_dv + S_omega * dvlin_dv;

      // wrt a:
      // no correction because omega and vlin do not depend on a.
    }

    acceleration.dfdx.setZero(3, stateConverter_->getStateVariableDim());
    acceleration.dfdx.leftCols(stateConverter_->getTangentDim()) = a_partial_dq.bottomRows(3);
    acceleration.dfdx.rightCols(stateConverter_->getTangentDim()) = a_partial_dv.bottomRows(3);

    acceleration.dfdu.setZero(3, stateConverter_->getInputDim());
    if (stateConverter_->getBaseVDim() > 0) {
      for (int i=0; i<stateConverter_->contactCandidateIds.size(); i++) {
        ocs2::matrix_t J = ocs2::matrix_t::Zero(6, stateConverter_->getTangentDim());
        pinocchio::getFrameJacobian(model, data, stateConverter_->contactCandidateIds[i], rf, J);
        acceleration.dfdu.block(0,i*6, 3, 6) = (data.M.topLeftCorner(6,6).inverse() * J.block(0,0,6,6).transpose()).bottomRows(3);
      }
    }
    acceleration.dfdu.rightCols(stateConverter_->getJointDim()) = (a_partial_da.rightCols(stateConverter_->getJointDim()) + a_partial_da.leftCols(stateConverter_->getBaseVDim()) * data.M.topLeftCorner(6,6).inverse() * data.M.block(0, 6, 6, stateConverter_->getJointDim())).bottomRows(3);
    return acceleration;
  }


  ocs2::vector_t PinocchioFrameDynamics::getAccelerations(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    vector6_t acceleration;
    acceleration.head(3) = pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).linear();
    acceleration.tail(3) = pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).angular();
    return acceleration;
  }

  ocs2::VectorFunctionLinearApproximation PinocchioFrameDynamics::getAccelerationsLinearApproximation(const OCPPreComputation& preComputation) const {
    ocs2::PinocchioInterface& pinocchioInterface = preComputation.getPinocchioInterface();
    const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();

    ocs2::VectorFunctionLinearApproximation acceleration;
    vector6_t acceleration_f;
    acceleration_f.head(3) = pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).linear();
    acceleration_f.tail(3) = pinocchio::getFrameClassicalAcceleration(model, data, frameId_, rf).angular();
    acceleration.f = acceleration_f;

    ocs2::matrix_t v_partial_dq(6, model.nv);
    ocs2::matrix_t spatial_dq(6, model.nv);
    ocs2::matrix_t spatial_dv(6, model.nv);
    ocs2::matrix_t spatial_da(6, model.nv);
    pinocchio::getFrameAccelerationDerivatives(model, data, frameId_, rf, v_partial_dq, spatial_dq, spatial_dv, spatial_da);

    ocs2::matrix_t a_partial_dq = spatial_dq;
    ocs2::matrix_t a_partial_dv = spatial_dv;
    ocs2::matrix_t a_partial_da = spatial_da;
    if (!a_partial_da.allFinite()) {
      a_partial_da = a_partial_da.unaryExpr([](double x) {
                                              return std::isfinite(x) ? x : 0.0;
                                            });
    }

    const pinocchio::Motion v_frame = pinocchio::getFrameVelocity(model, data, frameId_, rf);
    const vector3_t omega = v_frame.angular();
    const vector3_t vlin  = v_frame.linear();
    const ocs2::matrix_t S_omega = ocs2::skewSymmetricMatrix(omega);
    const ocs2::matrix_t S_vlin = ocs2::skewSymmetricMatrix(vlin);

    for (int k = 0; k < model.nv; ++k) {
      // wrt q
      const vector3_t dvlin_dq  = spatial_dq.block<3,1>(0, k);
      const vector3_t domega_dq = spatial_dq.block<3,1>(3, k);

      a_partial_dq.block<3,1>(0, k) +=
        -S_vlin * domega_dq + S_omega * dvlin_dq;

      // wrt v
      const vector3_t dvlin_dv  = spatial_dv.block<3,1>(0, k);
      const vector3_t domega_dv = spatial_dv.block<3,1>(3, k);

      a_partial_dv.block<3,1>(0, k) +=
        -S_vlin * domega_dv + S_omega * dvlin_dv;

      // wrt a:
      // no correction because omega and vlin do not depend on a.
    }

    acceleration.dfdx.setZero(6, stateConverter_->getStateVariableDim());
    acceleration.dfdx.leftCols(stateConverter_->getTangentDim()) = a_partial_dq;
    acceleration.dfdx.rightCols(stateConverter_->getTangentDim()) = a_partial_dv;

    acceleration.dfdu.setZero(6, stateConverter_->getInputDim());
    if (stateConverter_->getBaseVDim() > 0) {
      for (int i=0; i<stateConverter_->contactCandidateIds.size(); i++) {
        ocs2::matrix_t J = ocs2::matrix_t::Zero(6, stateConverter_->getTangentDim());
        pinocchio::getFrameJacobian(model, data, stateConverter_->contactCandidateIds[i], rf, J);
        acceleration.dfdu.block(0,i*6, 6, 6) = data.M.topLeftCorner(6,6).inverse() * J.block(0,0,6,6).transpose() + a_partial_da.leftCols(stateConverter_->getBaseVDim()) * data.M.topLeftCorner(6,6).inverse() * J.block(0,0,6,6).transpose();
      }
    }

    acceleration.dfdu.rightCols(stateConverter_->getJointDim()) = a_partial_da.rightCols(stateConverter_->getJointDim()) + a_partial_da.leftCols(stateConverter_->getBaseVDim()) * data.M.topLeftCorner(6,6).inverse() * data.M.block(0, 6, 6, stateConverter_->getJointDim());

    return acceleration;
  }

}
