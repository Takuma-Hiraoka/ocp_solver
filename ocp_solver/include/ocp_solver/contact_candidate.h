#pragma once
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocp_solver {
  class ContactCandidate {
  public:
    std::string name;
    pinocchio::SE3 localPose;
  };
}
