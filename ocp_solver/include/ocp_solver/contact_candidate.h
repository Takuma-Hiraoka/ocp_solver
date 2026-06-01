#pragma once
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>

#include <Eigen/Core>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace ocp_solver {
  using ContactCandidateIndex = std::size_t;

  class ContactCandidate {
  public:
    std::string frameName;
    std::string parentJointName;
    pinocchio::SE3 localPose;
    bool searchContactPoint = false;
    bool alignContactFrameWithMeshNormal = false;
    std::vector<Eigen::Vector3d> meshVerticesInLocalFrame;
    std::vector<Eigen::Vector3d> meshNormalsInLocalFrame;
    double meshNormalSubmeshRadius = std::numeric_limits<double>::infinity();
  };

  template <typename SCALAR_T>
  class ContactCandidateInfoTpl {
  public:
    ContactCandidateIndex index = 0;
    std::string frameName;
    pinocchio::JointIndex parentJointIndex = 0;
    pinocchio::SE3Tpl<SCALAR_T> localFramePose = pinocchio::SE3Tpl<SCALAR_T>::Identity();
    pinocchio::SE3Tpl<SCALAR_T> localPoseInLocalFrame = pinocchio::SE3Tpl<SCALAR_T>::Identity();
    pinocchio::SE3Tpl<SCALAR_T> localPose = pinocchio::SE3Tpl<SCALAR_T>::Identity();
    bool searchContactPoint = false;
    bool alignContactFrameWithMeshNormal = false;
    std::shared_ptr<const std::vector<Eigen::Vector3d>> meshVerticesInLocalFrame;
    std::shared_ptr<const std::vector<Eigen::Vector3d>> meshNormalsInLocalFrame;
    double meshNormalSubmeshRadius = std::numeric_limits<double>::infinity();
    size_t contactPointStateIndex = 0;
  };

  using ContactCandidateInfo = ContactCandidateInfoTpl<double>;
}
