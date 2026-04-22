#pragma once
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocp_solver {
  class ContactCandidate {
  public:
    std::string name;
    std::string parentJointName;
    pinocchio::SE3 localPose;
  };
}
