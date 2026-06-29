#include "ocp_constraint/cop_constraint.h"

#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>

#include <pinocchio/algorithm/jacobian.hpp>

namespace ocp_constraint {
namespace {

Eigen::Vector3d normalAlignedWithForce(const Eigen::Vector3d& normal,
                                       const Eigen::Vector3d& localForce) {
  return normal.dot(localForce) < 0.0 ? -normal : normal;
}

}  // namespace

CopConstraint::CopConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                             size_t contactIndex,
                             const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                             const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                             Config config)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      contactIndex_(contactIndex),
      config_(config),
      stateConverterPtr_(stateConverter.clone()),
      frameDynamicsPtr_(frameDynamics.clone()),
      referenceManagerPtr_(&referenceManager) {}

CopConstraint::CopConstraint(const CopConstraint& rhs)
    : StateInputConstraint(rhs),
      contactIndex_(rhs.contactIndex_),
      config_(rhs.config_),
      stateConverterPtr_(rhs.stateConverterPtr_->clone()),
      frameDynamicsPtr_(rhs.frameDynamicsPtr_->clone()),
      referenceManagerPtr_(rhs.referenceManagerPtr_) {}

bool CopConstraint::isActive(ocs2::scalar_t time) const {
  return referenceManagerPtr_->isInContact(time, stateConverterPtr_->getContactCandidateIds()[contactIndex_]);
}

ocs2::vector_t CopConstraint::getValue(ocs2::scalar_t /*time*/,
                                       const ocs2::vector_t& state,
                                       const ocs2::vector_t& input,
                                       const ocs2::PreComputation& preComp) const {
  const auto& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
  ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
  const auto contactCandidate = stateConverterPtr_->getContactCandidate(state, contactIndex_);
  const pinocchio::SE3 contactPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, contactCandidate);

  Eigen::Matrix<ocs2::scalar_t, 6, 6> worldToContact =
      Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
  worldToContact.block<3, 3>(0, 0) = contactPose.rotation().transpose();
  worldToContact.block<3, 3>(3, 3) = contactPose.rotation().transpose();
  const Eigen::Matrix<ocs2::scalar_t, 6, 1> localWrench =
      worldToContact * stateConverterPtr_->getContactWrench(input, contactIndex_);
  const Eigen::Vector3d forceNormal =
      normalAlignedWithForce(Eigen::Vector3d::UnitZ(), localWrench.head<3>());
  const ocs2::scalar_t normalForce =
      forceNormal.dot(localWrench.head<3>()) + config_.normalForceRegularization;
  const Eigen::Vector3d copLocal = forceNormal.cross(localWrench.tail<3>()) / normalForce;
  const Eigen::Vector3d copWorld = contactPose.translation() + contactPose.rotation() * copLocal;
  return copWorld - frameDynamicsPtr_->getPosition(ocpPreComp);
}

ocs2::VectorFunctionLinearApproximation CopConstraint::getLinearApproximation(
    ocs2::scalar_t time,
    const ocs2::vector_t& state,
    const ocs2::vector_t& input,
    const ocs2::PreComputation& preComp) const {
  const auto& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
  ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
  const auto contactCandidate = stateConverterPtr_->getContactCandidate(state, contactIndex_);
  const pinocchio::SE3 contactPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, contactCandidate);

  Eigen::Matrix<ocs2::scalar_t, 6, 6> worldToContact =
      Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
  worldToContact.block<3, 3>(0, 0) = contactPose.rotation().transpose();
  worldToContact.block<3, 3>(3, 3) = contactPose.rotation().transpose();
  const Eigen::Matrix<ocs2::scalar_t, 6, 1> worldWrench =
      stateConverterPtr_->getContactWrench(input, contactIndex_);
  const Eigen::Matrix<ocs2::scalar_t, 6, 1> localWrench = worldToContact * worldWrench;
  const Eigen::Vector3d localForce = localWrench.head<3>();
  const Eigen::Vector3d localMoment = localWrench.tail<3>();
  const Eigen::Vector3d forceNormal =
      normalAlignedWithForce(Eigen::Vector3d::UnitZ(), localForce);
  const ocs2::scalar_t normalForce =
      forceNormal.dot(localForce) + config_.normalForceRegularization;
  const Eigen::Vector3d copLocal = forceNormal.cross(localMoment) / normalForce;
  const Eigen::Vector3d copWorld = contactPose.translation() + contactPose.rotation() * copLocal;

  Eigen::Matrix<ocs2::scalar_t, 3, 6> dCopLocalDLocalWrench =
      Eigen::Matrix<ocs2::scalar_t, 3, 6>::Zero();
  dCopLocalDLocalWrench.leftCols<3>().noalias() =
      -(copLocal / normalForce) * forceNormal.transpose();
  dCopLocalDLocalWrench.rightCols<3>().noalias() =
      ocs2::skewSymmetricMatrix(forceNormal) / normalForce;

  ocs2::VectorFunctionLinearApproximation approx;
  approx.f = copWorld - frameDynamicsPtr_->getPosition(ocpPreComp);
  approx.dfdx = ocs2::matrix_t::Zero(3, stateConverterPtr_->getStateVariableDim());
  approx.dfdu = ocs2::matrix_t::Zero(3, stateConverterPtr_->getInputDim());
  approx.dfdu.block<3, 6>(0, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      contactPose.rotation() * dCopLocalDLocalWrench * worldToContact;

  const ocs2::VectorFunctionLinearApproximation referenceApprox =
      frameDynamicsPtr_->getPositionLinearApproximation(ocpPreComp);
  approx.dfdx.noalias() -= referenceApprox.dfdx;

  ocs2::matrix_t contactJacobian =
      ocs2::matrix_t::Zero(6, stateConverterPtr_->getTangentDim());
  ocp_solver::getContactCandidateJacobian(pinocchioInterface, contactCandidate,
                                          pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                          contactJacobian);

  approx.dfdx.leftCols(stateConverterPtr_->getTangentDim()).noalias() += contactJacobian.topRows(3);
  const Eigen::Vector3d contactToCopWorld = copWorld - contactPose.translation();
  approx.dfdx.leftCols(stateConverterPtr_->getTangentDim()).noalias() +=
      -ocs2::skewSymmetricMatrix(contactToCopWorld) * contactJacobian.bottomRows(3);

  Eigen::Matrix<ocs2::scalar_t, 6, 6> dLocalWrenchDFrameRotation =
      Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
  dLocalWrenchDFrameRotation.block<3, 3>(0, 3) =
      contactPose.rotation().transpose() * ocs2::skewSymmetricMatrix(Eigen::Vector3d(worldWrench.head<3>()));
  dLocalWrenchDFrameRotation.block<3, 3>(3, 3) =
      contactPose.rotation().transpose() * ocs2::skewSymmetricMatrix(Eigen::Vector3d(worldWrench.tail<3>()));
  approx.dfdx.leftCols(stateConverterPtr_->getTangentDim()).noalias() +=
      contactPose.rotation() * dCopLocalDLocalWrench * dLocalWrenchDFrameRotation * contactJacobian;

  return approx;
}

}  // namespace ocp_constraint
