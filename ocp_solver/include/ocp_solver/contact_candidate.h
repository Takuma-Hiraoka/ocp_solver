#pragma once
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocp_solver {
  using ContactCandidateIndex = std::size_t;

  class ContactCandidate {
  public:
    std::string frameName;
    std::string parentJointName;
    pinocchio::SE3 localPose;
  };

  template <typename SCALAR_T>
  class ContactCandidateInfoTpl {
  public:
    ContactCandidateIndex index = 0;
    std::string frameName;
    pinocchio::JointIndex parentJointIndex = 0;
    pinocchio::SE3Tpl<SCALAR_T> localPose = pinocchio::SE3Tpl<SCALAR_T>::Identity();
  };

  using ContactCandidateInfo = ContactCandidateInfoTpl<double>;
}
