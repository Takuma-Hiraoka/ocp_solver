#pragma once

#include <pinocchio/fwd.hpp>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocp_solver {
  template <typename SCALAR_T>
    inline const Eigen::Matrix<SCALAR_T, 3, 3>& getRotationMatrixLocalToWorld(const pinocchio::DataTpl<SCALAR_T>& data,
                                                                              const pinocchio::FrameIndex localFrameIndex) {
    return data.oMf[localFrameIndex].rotation();
  }

  template <typename SCALAR_T>
    inline Eigen::Matrix<SCALAR_T, 3, 1> rotateVectorWorldToLocal(const Eigen::Matrix<SCALAR_T, 3, 1>& vectorInWorldFrame,
                                                                  const pinocchio::DataTpl<SCALAR_T>& data,
                                                                  const pinocchio::FrameIndex& localFrameIndex) {
    const Eigen::Matrix<SCALAR_T, 3, 3>& R_WorldToLocal = getRotationMatrixLocalToWorld(data, localFrameIndex).transpose();
    return (R_WorldToLocal * vectorInWorldFrame);
  }

  template <typename SCALAR_T>
    inline Eigen::Matrix<SCALAR_T, 3, 1> rotateVectorLocalToWorld(const Eigen::Matrix<SCALAR_T, 3, 1>& vectorInLocalFrame,
                                                                  const pinocchio::DataTpl<SCALAR_T>& data,
                                                                  const pinocchio::FrameIndex& localFrameIndex) {
    const Eigen::Matrix<SCALAR_T, 3, 3>& R_WorldToLocal = getRotationMatrixLocalToWorld(data, localFrameIndex);
    return (R_WorldToLocal * vectorInLocalFrame);
  }

  template <typename SCALAR_T>
    inline Eigen::Matrix<SCALAR_T, 6, 1> rotateVectorWorldToLocal(const Eigen::Matrix<SCALAR_T, 6, 1>& vectorInWorldFrame,
                                                                  const pinocchio::DataTpl<SCALAR_T>& data,
                                                                  const pinocchio::FrameIndex& localFrameIndex) {
    Eigen::Matrix<SCALAR_T, 6, 1> vectorInLocalFrame(6);
    vectorInLocalFrame.head(3) = rotateVectorWorldToLocal(Eigen::Matrix<SCALAR_T, 3, 1>(vectorInWorldFrame.head(3)), data, localFrameIndex),
      vectorInLocalFrame.tail(3) = rotateVectorWorldToLocal(Eigen::Matrix<SCALAR_T, 3, 1>(vectorInWorldFrame.tail(3)), data, localFrameIndex);
    return vectorInLocalFrame;
  }

  template <typename SCALAR_T>
    inline Eigen::Matrix<SCALAR_T, 6, 1> rotateVectorLocalToWorld(const Eigen::Matrix<SCALAR_T, 6, 1>& vectorInLocalFrame,
                                                                  const pinocchio::DataTpl<SCALAR_T>& data,
                                                                  const pinocchio::FrameIndex& localFrameIndex) {
    Eigen::Matrix<SCALAR_T, 6, 1> vectorInWorldFrame(6);
    vectorInWorldFrame.head(3) = rotateVectorLocalToWorld(Eigen::Matrix<SCALAR_T, 3, 1>(vectorInLocalFrame.head(3)), data, localFrameIndex),
      vectorInWorldFrame.tail(3) = rotateVectorLocalToWorld(Eigen::Matrix<SCALAR_T, 3, 1>(vectorInLocalFrame.tail(3)), data, localFrameIndex);
    return vectorInWorldFrame;
  }

  template <typename SCALAR_T>
    inline Eigen::Matrix<SCALAR_T, 4, 4> getTransformationMatrixLocalToWorld(const pinocchio::DataTpl<SCALAR_T>& data,
                                                                             const pinocchio::FrameIndex localFrameIndex) {
    return (data.oMf[localFrameIndex].toHomogeneousMatrix_impl());
  }

  template <typename SCALAR_T>
    inline Eigen::Matrix<SCALAR_T, 4, 4> getTransformationMatrixWorldToLocal(const pinocchio::DataTpl<SCALAR_T>& data,
                                                                             const pinocchio::FrameIndex localFrameIndex) {
    Eigen::Matrix<SCALAR_T, 4, 4> transformWorldToLocal = Eigen::Matrix<SCALAR_T, 4, 4>::Identity();
    transformWorldToLocal.block(0, 0, 3, 3) = data.oMf[localFrameIndex].rotation().transpose();
    transformWorldToLocal.block(0, 3, 3, 1) = -data.oMf[localFrameIndex].rotation().transpose() * data.oMf[localFrameIndex].translation();
    return transformWorldToLocal;
  }

  template <typename SCALAR_T>
    inline Eigen::Matrix<SCALAR_T, 3, 1> transformPointLocalToWorld(const Eigen::Matrix<SCALAR_T, 3, 1>& pointInLocalFrame,
                                                                    const pinocchio::DataTpl<SCALAR_T>& data,
                                                                    const pinocchio::FrameIndex& localFrameIndex) {
    Eigen::Matrix<SCALAR_T, 4, 4> homogeniousPoint;
    homogeniousPoint << pointInLocalFrame, 1.0;
    return (getTransformationMatrixLocalToWorld(data, localFrameIndex) * homogeniousPoint).head(3);
  }

  template <typename SCALAR_T>
    inline Eigen::Matrix<SCALAR_T, 3, 1> transformPointWorldToLocal(const Eigen::Matrix<SCALAR_T, 3, 1>& pointInWorldFrame,
                                                                    const pinocchio::DataTpl<SCALAR_T>& data,
                                                                    const pinocchio::FrameIndex& localFrameIndex) {
    Eigen::Matrix<SCALAR_T, 4, 4> homogeniousPoint;
    homogeniousPoint << pointInWorldFrame, 1.0;
    return (getTransformationMatrixWorldToLocal(data, localFrameIndex) * homogeniousPoint).head(3);
  }
}
