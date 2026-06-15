#pragma once

#include <vector>

#include <ocs2_core/Types.h>
#include <pinocchio/spatial/se3.hpp>

namespace ocp_solver {

  struct TargetSE3Trajectory {
    TargetSE3Trajectory();
    explicit TargetSE3Trajectory(pinocchio::SE3 targetPose,
                              ocs2::vector_t targetTwist = ocs2::vector_t::Zero(6),
                              ocs2::vector_t targetAcc = ocs2::vector_t::Zero(6));
    TargetSE3Trajectory(ocs2::scalar_array_t timeTrajectory,
                     std::vector<pinocchio::SE3> poseTrajectory,
                     ocs2::vector_array_t twistTrajectory,
                     ocs2::vector_array_t accTrajectory,
                     bool interpolate = true);

    pinocchio::SE3 getTargetPose(ocs2::scalar_t time) const;
    ocs2::vector_t getTargetTwist(ocs2::scalar_t time) const;
    ocs2::vector_t getTargetAcc(ocs2::scalar_t time) const;

    ocs2::scalar_array_t timeTrajectory;
    std::vector<pinocchio::SE3> poseTrajectory;
    ocs2::vector_array_t twistTrajectory;
    ocs2::vector_array_t accTrajectory;
    bool interpolate = true;
  };

  struct WrenchTrajectory {
    WrenchTrajectory();
    explicit WrenchTrajectory(ocs2::vector_t targetWrench);
    WrenchTrajectory(ocs2::scalar_array_t timeTrajectory,
                     ocs2::vector_array_t wrenchTrajectory,
                     bool interpolate = true);

    ocs2::vector_t getTargetWrench(ocs2::scalar_t time) const;

    ocs2::scalar_array_t timeTrajectory;
    ocs2::vector_array_t wrenchTrajectory;
    bool interpolate = true;
  };

}
