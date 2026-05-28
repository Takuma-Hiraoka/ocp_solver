#include <pinocchio/fwd.hpp>

#include "ocp_solver/solver/dynamics_helper_functions.h"

#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <type_traits>

// Pinnochio
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocp_solver {

  template <typename SCALAR_T>
  pinocchio::SE3Tpl<SCALAR_T> getContactCandidatePlacement(
      const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
      const ContactCandidateInfoTpl<SCALAR_T>& contactCandidate) {
    return pinInterface.getData().oMi[contactCandidate.parentJointIndex] * contactCandidate.localPose;
  }
  template pinocchio::SE3Tpl<ocs2::scalar_t> getContactCandidatePlacement(
      const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
      const ContactCandidateInfoTpl<ocs2::scalar_t>& contactCandidate);
  template pinocchio::SE3Tpl<ocs2::ad_scalar_t> getContactCandidatePlacement(
      const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinInterface,
      const ContactCandidateInfoTpl<ocs2::ad_scalar_t>& contactCandidate);

  template <typename SCALAR_T, typename Matrix6xLike>
  void getContactCandidateJacobian(
      const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
      const ContactCandidateInfoTpl<SCALAR_T>& contactCandidate,
      pinocchio::ReferenceFrame referenceFrame,
      const Eigen::MatrixBase<Matrix6xLike>& jacobian) {
    using DataType = std::remove_const_t<std::remove_reference_t<decltype(pinInterface.getData())>>;
    auto& data = const_cast<DataType&>(pinInterface.getData());
    pinocchio::getFrameJacobian(pinInterface.getModel(), data, contactCandidate.parentJointIndex,
                                contactCandidate.localPose, referenceFrame,
                                PINOCCHIO_EIGEN_CONST_CAST(Matrix6xLike, jacobian));
  }
  template void getContactCandidateJacobian<ocs2::scalar_t, ocs2::matrix_t>(
      const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
      const ContactCandidateInfoTpl<ocs2::scalar_t>& contactCandidate,
      pinocchio::ReferenceFrame referenceFrame,
      const Eigen::MatrixBase<ocs2::matrix_t>& jacobian);
  template void getContactCandidateJacobian<ocs2::ad_scalar_t, ocs2::ad_matrix_t>(
      const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinInterface,
      const ContactCandidateInfoTpl<ocs2::ad_scalar_t>& contactCandidate,
      pinocchio::ReferenceFrame referenceFrame,
      const Eigen::MatrixBase<ocs2::ad_matrix_t>& jacobian);

  template <typename SCALAR_T, typename Matrix6xLike>
  void getContactCandidateJacobianTimeVariation(
      const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
      const ContactCandidateInfoTpl<SCALAR_T>& contactCandidate,
      pinocchio::ReferenceFrame referenceFrame,
      const Eigen::MatrixBase<Matrix6xLike>& jacobianTimeVariation) {
    using DataType = std::remove_const_t<std::remove_reference_t<decltype(pinInterface.getData())>>;
    using ModelType = std::remove_const_t<std::remove_reference_t<decltype(pinInterface.getModel())>>;
    using SE3 = typename DataType::SE3;
    using Vector3 = typename SE3::Vector3;
    using Motion = typename DataType::Motion;

    const ModelType& model = pinInterface.getModel();
    auto& data = const_cast<DataType&>(pinInterface.getData());
    Matrix6xLike& dJ = PINOCCHIO_EIGEN_CONST_CAST(Matrix6xLike, jacobianTimeVariation);

    const pinocchio::JointIndex jointId = contactCandidate.parentJointIndex;
    const SE3 oMframe = data.oMi[jointId] * contactCandidate.localPose;
    pinocchio::details::translateJointJacobian(model, data, jointId, referenceFrame, oMframe, data.dJ, dJ);

    switch (referenceFrame) {
      case pinocchio::LOCAL: {
        const Motion& vJoint = data.v[jointId];
        const Motion vFrame = contactCandidate.localPose.actInv(vJoint);
        const int colRef = pinocchio::nv(model.joints[jointId]) + pinocchio::idx_v(model.joints[jointId]) - 1;
        for (Eigen::Index j = colRef; j >= 0; j = data.parents_fromRow[static_cast<size_t>(j)]) {
          typedef typename DataType::Matrix6x::ColXpr ColXprIn;
          typedef const pinocchio::MotionRef<ColXprIn> MotionIn;
          typedef typename Matrix6xLike::ColXpr ColXprOut;
          typedef pinocchio::MotionRef<ColXprOut> MotionOut;
          MotionIn vIn(data.J.col(j));
          MotionOut vOut(dJ.col(j));
          vOut -= vFrame.cross(oMframe.actInv(vIn));
        }
        break;
      }
      case pinocchio::LOCAL_WORLD_ALIGNED: {
        const Motion& ovJoint = data.ov[jointId];
        const int colRef = pinocchio::nv(model.joints[jointId]) + pinocchio::idx_v(model.joints[jointId]) - 1;
        for (Eigen::Index j = colRef; j >= 0; j = data.parents_fromRow[static_cast<size_t>(j)]) {
          typedef typename DataType::Matrix6x::ColXpr ColXprIn;
          typedef const pinocchio::MotionRef<ColXprIn> MotionIn;
          typedef typename Matrix6xLike::ColXpr ColXprOut;
          typedef pinocchio::MotionRef<ColXprOut> MotionOut;
          MotionIn vIn(data.J.col(j));
          MotionOut vOut(dJ.col(j));
          vOut.linear() -= Vector3(ovJoint.linear() + ovJoint.angular().cross(oMframe.translation())).cross(vIn.angular());
        }
        break;
      }
      case pinocchio::WORLD:
      default:
        break;
    }
  }
  template void getContactCandidateJacobianTimeVariation<ocs2::scalar_t, ocs2::matrix_t>(
      const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
      const ContactCandidateInfoTpl<ocs2::scalar_t>& contactCandidate,
      pinocchio::ReferenceFrame referenceFrame,
      const Eigen::MatrixBase<ocs2::matrix_t>& jacobianTimeVariation);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, 6, 1> computeBaseAcceleration(const Eigen::Matrix<SCALAR_T, -1, -1>& M,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& nle,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& externalForcesInJointSpace) {
    Eigen::Matrix<SCALAR_T, 6, Eigen::Dynamic> M_bj = M.block(0, 6, 6, qdd_joints.size());
    Eigen::Matrix<SCALAR_T, 6, 1> intermediate = -nle.head(6) - M_bj * qdd_joints + externalForcesInJointSpace.head(6);
    if constexpr (std::is_same_v<SCALAR_T, ocs2::ad_scalar_t>) {
    // Due to the block diagonal structure of the generalized mass matrix corresponding to the base the base mass matrix can be split into a
    // linear and angular part. Which are both inverted separately. This does not only exploit part of the sparsity but also prevents a CppAD
    // branching error when multiplying a 6x6 matrix with a6 dim. vector.

      Eigen::Matrix<SCALAR_T, 3, 3> M_bb_lin = M.topLeftCorner(3, 3);
      Eigen::Matrix<SCALAR_T, 3, 3> M_bb_ang = M.block(3, 3, 3, 3);
      Eigen::Matrix<SCALAR_T, 3, 3> M_bb_lin_inv = M_bb_lin.inverse();
      Eigen::Matrix<SCALAR_T, 3, 3> M_bb_ang_inv = M_bb_ang.inverse();


      Eigen::Matrix<SCALAR_T, 6, 1> baseAccelerations;//
      baseAccelerations.head(3) = M_bb_lin_inv * intermediate.head(3);
      baseAccelerations.tail(3) = M_bb_ang_inv * intermediate.tail(3);

      return baseAccelerations;
    } else {
      Eigen::Matrix<SCALAR_T, 6, 1> baseAccelerations = M.topLeftCorner(6, 6).inverse() * intermediate;
      return baseAccelerations;
    }
  }
  template Eigen::Matrix<ocs2::scalar_t, 6, 1> computeBaseAcceleration(const Eigen::Matrix<ocs2::scalar_t, -1, -1>& M,
                                                                       const Eigen::Matrix<ocs2::scalar_t, -1, 1>& nle,
                                                                       const Eigen::Matrix<ocs2::scalar_t, -1, 1>& qdd_joints,
                                                                       const Eigen::Matrix<ocs2::scalar_t, -1, 1>& externalForcesInJointSpace);
  template Eigen::Matrix<ocs2::ad_scalar_t, 6, 1> computeBaseAcceleration(const Eigen::Matrix<ocs2::ad_scalar_t, -1, -1>& M,
                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& nle,
                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qdd_joints,
                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& externalForcesInJointSpace);

  BaseAccelerationLinearApproximation computeBaseAccelerationLinearApproximation(
    const ocs2::vector_t& q,
    const ocs2::vector_t& v,
    const ocs2::vector_t& generalizedAccelerations,
    const ocs2::vector_t& input,
    ocs2::PinocchioInterface& pinocchioInterface,
    const StateConverter<ocs2::scalar_t>& stateConverter,
    pinocchio::ReferenceFrame referenceFrame) {
    const pinocchio::Model& model = pinocchioInterface.getModel();
    pinocchio::Data& data = pinocchioInterface.getData();
    const size_t tangentDim = stateConverter.getTangentDim();
    const size_t baseVDim = stateConverter.getBaseVDim();
    const size_t jointDim = stateConverter.getJointDim();

    BaseAccelerationLinearApproximation approximation;
    approximation.dfdq.setZero(baseVDim, tangentDim);
    approximation.dfdv.setZero(baseVDim, tangentDim);
    approximation.dfdu.setZero(baseVDim, stateConverter.getInputDim());
    if (baseVDim == 0) {
      return approximation;
    }

    pinocchio::forwardKinematics(model, data, q, v, generalizedAccelerations);
    pinocchio::computeRNEADerivatives(model, data, q, v, generalizedAccelerations);
    const ocs2::matrix_t dtau_dq = data.dtau_dq;
    const ocs2::matrix_t dtau_dv = data.dtau_dv;
    ocs2::matrix_t M = data.M;
    M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();

    ocs2::matrix_t externalForcesDerivative = ocs2::matrix_t::Zero(tangentDim, tangentDim);
    for (size_t column = 0; column < tangentDim; ++column) {
      ocs2::vector_t tangentDirection = ocs2::vector_t::Zero(tangentDim);
      tangentDirection(column) = 1.0;
      pinocchio::computeJointJacobiansTimeVariation(model, data, q, tangentDirection);

      for (size_t i = 0; i < stateConverter.contactCandidates.size(); ++i) {
        ocs2::matrix_t dJ = ocs2::matrix_t::Zero(6, tangentDim);
        getContactCandidateJacobianTimeVariation(pinocchioInterface, stateConverter.getContactCandidate(i), referenceFrame, dJ);
        externalForcesDerivative.col(column).noalias() += dJ.transpose() * stateConverter.getContactWrench(input, i);
      }
    }

    const ocs2::matrix_t M_bb = M.topLeftCorner(baseVDim, baseVDim);
    const Eigen::LDLT<ocs2::matrix_t> MbbSolver = M_bb.ldlt();
    const ocs2::matrix_t baseResidualDerivativeQ = dtau_dq.topRows(baseVDim) - externalForcesDerivative.topRows(baseVDim);
    approximation.dfdq = -MbbSolver.solve(baseResidualDerivativeQ);
    approximation.dfdv = -MbbSolver.solve(dtau_dv.topRows(baseVDim));

    pinocchio::computeJointJacobians(model, data, q);
    for (size_t i = 0; i < stateConverter.contactCandidates.size(); ++i) {
      ocs2::matrix_t J = ocs2::matrix_t::Zero(6, tangentDim);
      getContactCandidateJacobian(pinocchioInterface, stateConverter.getContactCandidate(i), referenceFrame, J);
      approximation.dfdu.block(0, i * 6, baseVDim, 6) = MbbSolver.solve(J.block(0, 0, 6, baseVDim).transpose());
    }

    approximation.dfdu.block(0, stateConverter.getJointAccelerationsStartindex(), baseVDim, jointDim) =
      -MbbSolver.solve(M.block(0, baseVDim, baseVDim, jointDim));
    return approximation;
  }

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, 6, 1> computeBaseAcceleration(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                        const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                        StateConverter<SCALAR_T>& stateConverter) {
    auto& pinInterfaceNonConst = const_cast<ocs2::PinocchioInterfaceTpl<SCALAR_T>&>(pinInterface);
    const pinocchio::ModelTpl<SCALAR_T>& model = pinInterface.getModel();
    pinocchio::DataTpl<SCALAR_T>& data = pinInterfaceNonConst.getData();
    const Eigen::Matrix<SCALAR_T, -1, 1> q = stateConverter.getGeneralizedCoordinates(state);
    const Eigen::Matrix<SCALAR_T, -1, 1> qd = stateConverter.getGeneralizedVelocities(state, input);
    const Eigen::Matrix<SCALAR_T, -1, 1> qdd_joints = stateConverter.getJointAccelerations(input);

    data.M.fill(SCALAR_T(0.0));
    pinocchio::forwardKinematics(model, data, q, qd);
    pinocchio::crba(model, data, q);
    pinocchio::nonLinearEffects(model, data, q, qd);

    Eigen::Matrix<SCALAR_T, 6, 1> baseExternalForces = Eigen::Matrix<SCALAR_T, 6, 1>::Zero();

    pinocchio::computeJointJacobians(model, data, q);
    for (int i=0; i<stateConverter.contactCandidates.size(); i++) {
      Eigen::Matrix<SCALAR_T, -1, -1> J = Eigen::Matrix<SCALAR_T, -1, -1>::Zero(6, stateConverter.getTangentDim());
      getContactCandidateJacobian(pinInterfaceNonConst, stateConverter.getContactCandidate(i), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
      Eigen::Matrix<SCALAR_T, 6, 6> J_b = J.block(0, 0, 6, 6);
      baseExternalForces += J_b.transpose() * stateConverter.getContactWrench(input, i);
    }

    return computeBaseAcceleration<SCALAR_T>(data.M, data.nle, qdd_joints, baseExternalForces);
  }
  template Eigen::Matrix<ocs2::ad_scalar_t, 6, 1> computeBaseAcceleration(const ocs2::ad_vector_t& state,
                                                                        const ocs2::ad_vector_t& input,
                                                                        const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinInterface,
                                                                        StateConverter<ocs2::ad_scalar_t>& stateConverter);
  template Eigen::Matrix<ocs2::scalar_t, 6, 1> computeBaseAcceleration(const ocs2::vector_t& state,
                                                                        const ocs2::vector_t& input,
                                                                        const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
                                                                        StateConverter<ocs2::scalar_t>& stateConverter);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeGeneralizedAccelerations(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                                 const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                                 const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                                 StateConverter<SCALAR_T>& stateConverter) {
    // Generalized Accelerations = [ddq_base, ddq_joints]
    Eigen::Matrix<SCALAR_T, -1, 1> generalizedAccelerations = Eigen::Matrix<SCALAR_T, -1, 1>::Zero(stateConverter.getTangentDim());
    if (stateConverter.getBaseVDim() == 6) generalizedAccelerations.head(6) = computeBaseAcceleration<SCALAR_T>(state, input, pinInterface, stateConverter);
    generalizedAccelerations.segment(stateConverter.getBaseVDim(), stateConverter.getJointDim()) = stateConverter.getJointAccelerations(input);
    return generalizedAccelerations;
  }
  template ocs2::ad_vector_t computeGeneralizedAccelerations(const ocs2::ad_vector_t& state,
                                                             const ocs2::ad_vector_t& input,
                                                             const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinInterface,
                                                             StateConverter<ocs2::ad_scalar_t>& stateConverter);
  template ocs2::vector_t computeGeneralizedAccelerations(const ocs2::vector_t& state,
                                                          const ocs2::vector_t& input,
                                                          const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
                                                          StateConverter<ocs2::scalar_t>& stateConverter);


  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeStateDerivative(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                        const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                        const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                        StateConverter<SCALAR_T>& stateConverter) {
    // State derivative = [dq; ddq_base, ddq_joints]
    Eigen::Matrix<SCALAR_T, -1, 1> state_derivative = Eigen::Matrix<SCALAR_T, -1, 1>::Zero(stateConverter.getStateVariableDim());
    if (stateConverter.getBaseVDim() == 6) {
      state_derivative.head(3) = state.segment(stateConverter.getGenCoordinatesDim(), 3);
      // Derivatives of the euler angles ZYX
      state_derivative.segment(3, 3) = state.segment(stateConverter.getGenCoordinatesDim() + 3, 3);
    }
    state_derivative.segment(stateConverter.getBaseVDim(), stateConverter.getJointDim()) = stateConverter.getJointVelocities(state, input);
    state_derivative.segment(stateConverter.getTangentDim(), stateConverter.getTangentDim()) =
      computeGeneralizedAccelerations<SCALAR_T>(state, input, pinInterface, stateConverter);
    return state_derivative;
  }
  template ocs2::ad_vector_t computeStateDerivative(const ocs2::ad_vector_t& state,
                                                    const ocs2::ad_vector_t& input,
                                                    const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinInterface,
                                                    StateConverter<ocs2::ad_scalar_t>& stateConverter);
  template ocs2::vector_t computeStateDerivative(const ocs2::vector_t& state,
                                                 const ocs2::vector_t& input,
                                                 const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
                                                 StateConverter<ocs2::scalar_t>& stateConverter);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorques(const Eigen::Matrix<SCALAR_T, -1, 1>& state,
                                                     const Eigen::Matrix<SCALAR_T, -1, 1>& input,
                                                     const ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinInterface,
                                                     StateConverter<SCALAR_T>& stateConverter) {
    const Eigen::Matrix<SCALAR_T, -1, 1> q = stateConverter.getGeneralizedCoordinates(state);
    const Eigen::Matrix<SCALAR_T, -1, 1> qd = stateConverter.getGeneralizedVelocities(state, input);
    const Eigen::Matrix<SCALAR_T, -1, 1> qdd_joints = stateConverter.getJointAccelerations(input);

    std::vector<std::pair<Eigen::Matrix<SCALAR_T, 6, 1>, ContactCandidateInfoTpl<SCALAR_T>>> wrenches(
      stateConverter.contactCandidates.size(), {Eigen::Matrix<SCALAR_T, 6, 1>::Zero(), ContactCandidateInfoTpl<SCALAR_T>()});
    for (int i=0; i<wrenches.size(); i++) {
      wrenches[i].first = stateConverter.getContactWrench(input, i);
      wrenches[i].second = stateConverter.getContactCandidate(i);
    }

    auto& pinInterfaceNonConst = const_cast<ocs2::PinocchioInterfaceTpl<SCALAR_T>&>(pinInterface);
    return computeJointTorques<SCALAR_T>(q, qd, qdd_joints, wrenches, pinInterfaceNonConst);
  }
  template ocs2::ad_vector_t computeJointTorques(const ocs2::ad_vector_t& state,
                                                 const ocs2::ad_vector_t& input,
                                                 const ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinInterface,
                                                 StateConverter<ocs2::ad_scalar_t>& stateConverter);
  template ocs2::vector_t computeJointTorques(const ocs2::vector_t& state,
                                              const ocs2::vector_t& input,
                                              const ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinInterface,
                                              StateConverter<ocs2::scalar_t>& stateConverter);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorques(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                                     const Eigen::Matrix<SCALAR_T, -1, 1>& qd,
                                                     const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                     const std::vector<std::pair<Eigen::Matrix<SCALAR_T, 6, 1>, ContactCandidateInfoTpl<SCALAR_T>>>& wrenches,
                                                     ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
    const pinocchio::ModelTpl<SCALAR_T>& model = pinocchioInterface.getModel();
    pinocchio::DataTpl<SCALAR_T>& data = pinocchioInterface.getData();

    pinocchio::crba(model, data, q);
    pinocchio::nonLinearEffects(model, data, q, qd);

    size_t n_qd = qd.size();
    Eigen::Matrix<SCALAR_T, -1, 1> externalForcesInJointSpace = Eigen::Matrix<SCALAR_T, -1, 1>::Zero(n_qd);

    for (int i=0; i<wrenches.size(); i++) {
      Eigen::Matrix<SCALAR_T, -1, -1> J = Eigen::Matrix<SCALAR_T, -1, -1>::Zero(6, n_qd);
      pinocchio::computeJointJacobians(model, data, q);
      getContactCandidateJacobian(pinocchioInterface, wrenches[i].second, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
      externalForcesInJointSpace += J.transpose() * wrenches[i].first;
    }

    Eigen::Matrix<SCALAR_T, -1, 1> q_dd(n_qd);
    if (n_qd > qdd_joints.size()) q_dd.head(6) = computeBaseAcceleration(data.M, data.nle, qdd_joints, externalForcesInJointSpace);
    q_dd.segment(n_qd - qdd_joints.size(), qdd_joints.size()) = qdd_joints;

    size_t n_joints = qdd_joints.size();

    Eigen::Matrix<SCALAR_T, -1, 1> jointTorques =
      data.M.block(n_qd - n_joints, 0, n_joints, n_qd) * q_dd +
      data.nle.segment(n_qd - n_joints, n_joints) -
      externalForcesInJointSpace.segment(n_qd - n_joints, n_joints);

    // return jointTorques;
    return jointTorques;
  }
  template Eigen::Matrix<ocs2::scalar_t, -1, 1> computeJointTorques(const Eigen::Matrix<ocs2::scalar_t, -1, 1>& q,
                                                                    const Eigen::Matrix<ocs2::scalar_t, -1, 1>& qd,
                                                                    const Eigen::Matrix<ocs2::scalar_t, -1, 1>& qdd_joints,
                                                                    const std::vector<std::pair<Eigen::Matrix<ocs2::scalar_t, 6, 1>, ContactCandidateInfoTpl<ocs2::scalar_t>>>& wrenches,
                                                                    ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface);
  template Eigen::Matrix<ocs2::ad_scalar_t, -1, 1> computeJointTorques(const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& q,
                                                                       const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qd,
                                                                       const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qdd_joints,
                                                                       const std::vector<std::pair<Eigen::Matrix<ocs2::ad_scalar_t, 6, 1>, ContactCandidateInfoTpl<ocs2::ad_scalar_t>>>& wrenches,
                                                                       ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface);

  template <typename SCALAR_T>
  Eigen::Matrix<SCALAR_T, -1, 1> computeJointTorquesRNEA(const Eigen::Matrix<SCALAR_T, -1, 1>& q,
                                                         const Eigen::Matrix<SCALAR_T, -1, 1>& qd,
                                                         const Eigen::Matrix<SCALAR_T, -1, 1>& qdd_joints,
                                                         const std::vector<std::pair<Eigen::Matrix<SCALAR_T, 6, 1>, ContactCandidateInfoTpl<SCALAR_T>>>& wrenches,
                                                         ocs2::PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
    const pinocchio::ModelTpl<SCALAR_T>& model = pinocchioInterface.getModel();
    pinocchio::DataTpl<SCALAR_T>& data = pinocchioInterface.getData();

    pinocchio::container::aligned_vector<pinocchio::Force> fextDesired(model.njoints, pinocchio::Force::Zero());

    pinocchio::forwardKinematics(model, data, q, qd);

    std::function<void(const ContactCandidateInfoTpl<SCALAR_T>&, size_t)> setExternalForce = [&](const ContactCandidateInfoTpl<SCALAR_T>& contactCandidate, size_t i) {
                              const pinocchio::JointIndex jointIndex = contactCandidate.parentJointIndex;
                              const Eigen::Matrix<SCALAR_T, 3, 1> translationJointFrameToContactFrame = contactCandidate.localPose.translation();
                              const Eigen::Matrix<SCALAR_T, 3, 3> rotationWorldFrameToJointFrame = data.oMi[jointIndex].rotation().transpose();
                              const Eigen::Matrix<SCALAR_T, 3, 1> contactForce = rotationWorldFrameToJointFrame * wrenches[i].first.head(3);
                              const Eigen::Matrix<SCALAR_T, 3, 1> contactTorque = rotationWorldFrameToJointFrame * wrenches[i].first.tail(3);
                              fextDesired[jointIndex].linear() = contactForce;
                              fextDesired[jointIndex].angular() = translationJointFrameToContactFrame.cross(contactForce) + contactTorque;
                            };

    for (int i=0; i<wrenches.size(); i++) setExternalForce(wrenches[i].second, i);

    pinocchio::crba(model, data, q);
    pinocchio::nonLinearEffects(model, data, q, qd);

    size_t n_qd = qd.size();
    Eigen::Matrix<SCALAR_T, -1, 1> externalForcesInJointSpace = Eigen::Matrix<SCALAR_T, -1, 1>::Zero(n_qd);

    for (int i=0; i<wrenches.size(); i++) {
      Eigen::Matrix<SCALAR_T, -1, -1> J = Eigen::Matrix<SCALAR_T, -1, -1>::Zero(6, n_qd);
      pinocchio::computeJointJacobians(model, data, q);
      getContactCandidateJacobian(pinocchioInterface, wrenches[i].second, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J);
      externalForcesInJointSpace += J.transpose() * wrenches[i].first;
    }

    // Repalce q with external forces in joint space.

    Eigen::Matrix<SCALAR_T, -1, 1> q_dd(n_qd);
    if (n_qd > qdd_joints.size()) q_dd.head(6) = computeBaseAcceleration(data.M, data.nle, qdd_joints, externalForcesInJointSpace);
    q_dd.segment(n_qd - qdd_joints.size(), qdd_joints.size()) = qdd_joints;

    ocs2::vector_t torques = pinocchio::rnea(model, data, q, qd, q_dd, fextDesired);

    return torques.segment(n_qd - qdd_joints.size(), qdd_joints.size());
  }
  template Eigen::Matrix<ocs2::scalar_t, -1, 1> computeJointTorquesRNEA(const Eigen::Matrix<ocs2::scalar_t, -1, 1>& q,
                                                                        const Eigen::Matrix<ocs2::scalar_t, -1, 1>& qd,
                                                                        const Eigen::Matrix<ocs2::scalar_t, -1, 1>& qdd_joints,
                                                                        const std::vector<std::pair<Eigen::Matrix<ocs2::scalar_t, 6, 1>, ContactCandidateInfoTpl<ocs2::scalar_t>>>& wrenches,
                                                                        ocs2::PinocchioInterfaceTpl<ocs2::scalar_t>& pinocchioInterface);
  // template Eigen::Matrix<ocs2::ad_scalar_t, -1, 1> computeJointTorquesRNEA(const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& q,
  //                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qd,
  //                                                                          const Eigen::Matrix<ocs2::ad_scalar_t, -1, 1>& qdd_joints,
  //                                                                          const std::vector<std::pair<Eigen::Matrix<ocs2::ad_scalar_t, 6, 1>, pinocchio::FrameIndex>>& wrenches,
  //                                                                          ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface);

}
