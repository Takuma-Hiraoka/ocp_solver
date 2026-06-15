#include <ocp_solver/solver/trajectory.h>

#include <algorithm>
#include <cassert>
#include <utility>

#include <pinocchio/spatial/explog.hpp>

namespace ocp_solver {
  namespace {
    std::pair<size_t, size_t> getInterpolationIndices(const ocs2::scalar_array_t& timeTrajectory,
                                                      ocs2::scalar_t time) {
      assert(!timeTrajectory.empty());
      if (time <= timeTrajectory.front()) return {0, 0};
      if (time >= timeTrajectory.back()) return {timeTrajectory.size() - 1, timeTrajectory.size() - 1};

      const auto upper = std::upper_bound(timeTrajectory.begin(), timeTrajectory.end(), time);
      const size_t upperIndex = static_cast<size_t>(std::distance(timeTrajectory.begin(), upper));
      return {upperIndex - 1, upperIndex};
    }

    ocs2::scalar_t interpolationRatio(const ocs2::scalar_array_t& timeTrajectory,
                                      size_t lowerIndex,
                                      size_t upperIndex,
                                      ocs2::scalar_t time) {
      if (lowerIndex == upperIndex) return 0.0;
      const ocs2::scalar_t duration = timeTrajectory[upperIndex] - timeTrajectory[lowerIndex];
      return duration > 0.0 ? (time - timeTrajectory[lowerIndex]) / duration : 0.0;
    }

    pinocchio::SE3 interpolatePose(const pinocchio::SE3& pose0,
                                   const pinocchio::SE3& pose1,
                                   ocs2::scalar_t alpha) {
      return pose0 * pinocchio::exp6(alpha * pinocchio::log6(pose0.inverse() * pose1));
    }

    ocs2::vector_t interpolateVector(const ocs2::vector_t& vector0,
                                     const ocs2::vector_t& vector1,
                                     ocs2::scalar_t alpha) {
      assert(vector0.size() == vector1.size());
      return (1.0 - alpha) * vector0 + alpha * vector1;
    }
  }

  TargetSE3Trajectory::TargetSE3Trajectory()
    : TargetSE3Trajectory(pinocchio::SE3::Identity()) {}

  TargetSE3Trajectory::TargetSE3Trajectory(pinocchio::SE3 targetPose,
                                     ocs2::vector_t targetTwist,
                                     ocs2::vector_t targetAcc)
    : timeTrajectory{0.0},
      poseTrajectory{std::move(targetPose)},
      twistTrajectory{std::move(targetTwist)},
      accTrajectory{std::move(targetAcc)},
      interpolate(false) {}

  TargetSE3Trajectory::TargetSE3Trajectory(ocs2::scalar_array_t timeTrajectory_,
                                     std::vector<pinocchio::SE3> poseTrajectory_,
                                     ocs2::vector_array_t twistTrajectory_,
                                     ocs2::vector_array_t accTrajectory_,
                                     bool interpolate_)
    : timeTrajectory(std::move(timeTrajectory_)),
      poseTrajectory(std::move(poseTrajectory_)),
      twistTrajectory(std::move(twistTrajectory_)),
      accTrajectory(std::move(accTrajectory_)),
      interpolate(interpolate_) {
    assert(!timeTrajectory.empty());
    assert(timeTrajectory.size() == poseTrajectory.size());
    assert(timeTrajectory.size() == twistTrajectory.size());
    assert(timeTrajectory.size() == accTrajectory.size());
  }

  pinocchio::SE3 TargetSE3Trajectory::getTargetPose(ocs2::scalar_t time) const {
    const auto [lowerIndex, upperIndex] = getInterpolationIndices(timeTrajectory, time);
    if ((lowerIndex == upperIndex) || !interpolate) {
      return poseTrajectory[lowerIndex];
    }
    return interpolatePose(poseTrajectory[lowerIndex],
                           poseTrajectory[upperIndex],
                           interpolationRatio(timeTrajectory, lowerIndex, upperIndex, time));
  }

  ocs2::vector_t TargetSE3Trajectory::getTargetTwist(ocs2::scalar_t time) const {
    const auto [lowerIndex, upperIndex] = getInterpolationIndices(timeTrajectory, time);
    if ((lowerIndex == upperIndex) || !interpolate) {
      return twistTrajectory[lowerIndex];
    }
    return interpolateVector(twistTrajectory[lowerIndex],
                             twistTrajectory[upperIndex],
                             interpolationRatio(timeTrajectory, lowerIndex, upperIndex, time));
  }

  ocs2::vector_t TargetSE3Trajectory::getTargetAcc(ocs2::scalar_t time) const {
    const auto [lowerIndex, upperIndex] = getInterpolationIndices(timeTrajectory, time);
    if ((lowerIndex == upperIndex) || !interpolate) {
      return accTrajectory[lowerIndex];
    }
    return interpolateVector(accTrajectory[lowerIndex],
                             accTrajectory[upperIndex],
                             interpolationRatio(timeTrajectory, lowerIndex, upperIndex, time));
  }

}
