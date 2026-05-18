#include "ocp_solver/system_dynamics.h"
#include "ocp_solver/dynamics_helper_functions.h"
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>

namespace ocp_solver {
  // parameterにqを入れる
  SystemDynamics::SystemDynamics(const ocs2::PinocchioInterface& pinocchioInterface,
                                 StateConverter<ocs2::scalar_t>& stateConverter)
    : SystemDynamicsBase(),
      pinInterface_(pinocchioInterface),
      stateConverter_(stateConverter) {}

  ocs2::vector_t SystemDynamics::computeFlowMap(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::PreComputation& preComputation) {
    return computeStateDerivative<ocs2::scalar_t>(x, u, pinInterface_, stateConverter_);
  }

  ocs2::vector_t SystemDynamics::computeJumpMap(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::PreComputation& preComputation) {
    return x;
  }

  ocs2::VectorFunctionLinearApproximation SystemDynamics::linearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u,
                                                                              const ocs2::PreComputation& preComputation) {

    ocs2::vector_t tau_eff = ocs2::vector_t::Zero(stateConverter_.getTangentDim());

    pinocchio::crba(pinInterface_.getModel(), pinInterface_.getData(), x.head(stateConverter_.getGenCoordinatesDim()));
    pinocchio::forwardKinematics(pinInterface_.getModel(), pinInterface_.getData(), x.head(stateConverter_.getGenCoordinatesDim()), x.tail(stateConverter_.getTangentDim()));
    pinocchio::updateFramePlacements(pinInterface_.getModel(), pinInterface_.getData());

    pinocchio::container::aligned_vector<pinocchio::Force> fextDesired(pinInterface_.getModel().njoints, pinocchio::Force::Zero());
    auto setExternalForce = [&](const pinocchio::FrameIndex& frameIndex, size_t i) {
                              const auto jointIndex = pinInterface_.getModel().frames[frameIndex].parentJoint;
                              const Eigen::Matrix<ocs2::scalar_t, 3, 1> translationJointFrameToContactFrame = pinInterface_.getModel().frames[frameIndex].placement.translation();
                              const Eigen::Matrix<ocs2::scalar_t, 3, 3> rotationWorldFrameToJointFrame = pinInterface_.getData().oMi[jointIndex].rotation().transpose();
                              const Eigen::Matrix<ocs2::scalar_t, 3, 1> contactForce = rotationWorldFrameToJointFrame * stateConverter_.getContactWrench(u, i).head(3);
                              const Eigen::Matrix<ocs2::scalar_t, 3, 1> contactTorque = rotationWorldFrameToJointFrame * stateConverter_.getContactWrench(u, i).tail(3);
                              fextDesired[jointIndex].linear() = contactForce;
                              fextDesired[jointIndex].angular() = translationJointFrameToContactFrame.cross(contactForce) + contactTorque;
                            };
    for (int i=0; i<stateConverter_.contactCandidateIds.size(); i++) setExternalForce(stateConverter_.contactCandidateIds[i], i);

    pinocchio::aba(pinInterface_.getModel(), pinInterface_.getData(), x.head(stateConverter_.getGenCoordinatesDim()), x.tail(stateConverter_.getTangentDim()), tau_eff, fextDesired);
    pinocchio::computeABADerivatives(pinInterface_.getModel(), pinInterface_.getData(), x.head(stateConverter_.getGenCoordinatesDim()), x.tail(stateConverter_.getTangentDim()), tau_eff, fextDesired);
    ocs2::matrix_t M_bj = pinInterface_.getData().M.block(0, 6, 6, stateConverter_.getJointDim());

    pinocchio::computeJointJacobians(pinInterface_.getModel(), pinInterface_.getData(), x.head(stateConverter_.getGenCoordinatesDim()));

    ocs2::VectorFunctionLinearApproximation approximation;
    approximation.dfdx.setZero(stateConverter_.getStateVariableDim(), stateConverter_.getStateVariableDim());
    approximation.dfdx.topRightCorner(stateConverter_.getTangentDim(), stateConverter_.getTangentDim()) = ocs2::matrix_t::Identity(stateConverter_.getTangentDim(), stateConverter_.getTangentDim());
    if (stateConverter_.getBaseVDim() == 6) {
      approximation.dfdx.block(stateConverter_.getTangentDim(), 0, 6, stateConverter_.getTangentDim()) = pinInterface_.getData().ddq_dq.topRows(6);
      approximation.dfdx.block(stateConverter_.getTangentDim(), stateConverter_.getTangentDim(), 6, stateConverter_.getTangentDim()) = pinInterface_.getData().ddq_dv.topRows(6);
    }

    approximation.dfdu.setZero(stateConverter_.getStateVariableDim(), u.rows());
    if (stateConverter_.getBaseVDim() == 6) {
      for (int i=0; i<stateConverter_.contactCandidateIds.size(); i++) {
        ocs2::matrix_t J = ocs2::matrix_t::Zero(6, stateConverter_.getTangentDim());
        pinocchio::getFrameJacobian(pinInterface_.getModel(), pinInterface_.getData(), stateConverter_.contactCandidateIds[i], pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
        approximation.dfdu.block(stateConverter_.getTangentDim(), i * 6, 6, 6) = pinInterface_.getData().M.topLeftCorner(6, 6).inverse() * J.block(0, 0, 6, 6).transpose();
      }
    }

    if (stateConverter_.getBaseVDim() == 6) {
      approximation.dfdu.block(stateConverter_.getTangentDim(), stateConverter_.getJointAccelerationsStartindex(), 6, stateConverter_.getJointDim()) = - pinInterface_.getData().M.topLeftCorner(6, 6).inverse() * M_bj;
    }
    approximation.dfdu.bottomRightCorner(stateConverter_.getJointDim(), stateConverter_.getJointDim()) = ocs2::matrix_t::Identity(stateConverter_.getJointDim(), stateConverter_.getJointDim());
    approximation.f = computeStateDerivative<ocs2::scalar_t>(x, u, pinInterface_, stateConverter_);
    return approximation;
  }

  ocs2::VectorFunctionLinearApproximation SystemDynamics::jumpMapLinearApproximation(ocs2::scalar_t t, const ocs2::vector_t& x,
                                                                                     const ocs2::PreComputation& preComputation) {
    ocs2::VectorFunctionLinearApproximation approximation;
    approximation.dfdx = ocs2::matrix_t::Identity(stateConverter_.getStateVariableDim(), stateConverter_.getStateVariableDim());
    approximation.dfdu.setZero(x.rows(), 0);
    approximation.f = x;
    return approximation;
  }
}
