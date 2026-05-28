#pragma once

#include <functional>
#include <string>
#include <vector>

#include <ocs2_core/Types.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "ocp_solver/solver/state_converter.h"
#include "ocp_solver/solver/ocp_pre_computation.h"

namespace ocp_solver {

  class PinocchioFrameDynamics {
  public:
    using vector3_t = Eigen::Matrix<ocs2::scalar_t, 3, 1>;
    using matrix3x_t = Eigen::Matrix<ocs2::scalar_t, 3, Eigen::Dynamic>;
    using vector6_t = Eigen::Matrix<ocs2::scalar_t, 6, 1>;
    using matrix6x_t = Eigen::Matrix<ocs2::scalar_t, 6, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<ocs2::scalar_t, Eigen::Dynamic, 1>;
    using quaternion_t = Eigen::Quaternion<ocs2::scalar_t>;

    PinocchioFrameDynamics(const ocs2::PinocchioInterface& pinocchioInterface,
                                 StateConverter<ocs2::scalar_t>& stateConverter,
                                 std::string frameName);
    PinocchioFrameDynamics(StateConverter<ocs2::scalar_t>& stateConverter, size_t contactIndex);

    ~PinocchioFrameDynamics() = default;
    PinocchioFrameDynamics* clone() const;
    PinocchioFrameDynamics& operator=(const PinocchioFrameDynamics&) = delete;

    const std::string& getId() const { return frameName_; };
    const std::size_t& getFrameId() const { return frameId_; };

    ocs2::vector_t getPosition(const OCPPreComputation& preComputation) const;
    ocs2::vector_t getVelocity(const OCPPreComputation& preComputation) const;

    quaternion_t getOrientation(const OCPPreComputation& preComputation) const;
    ocs2::vector_t getOrientationError(const OCPPreComputation& preComputation, const quaternion_t& referenceOrientation) const;
    ocs2::vector_t getAngularVelocity(const OCPPreComputation& preComputation) const;
    ocs2::vector_t getTwist(const OCPPreComputation& preComputation) const;
    ocs2::vector_t getLinearAcceleration(const OCPPreComputation& preComputation) const;
    ocs2::vector_t getAngularAcceleration(const OCPPreComputation& preComputation) const;
    ocs2::vector_t getAccelerations(const OCPPreComputation& preComputation) const;

    ocs2::VectorFunctionLinearApproximation getPositionLinearApproximation(const OCPPreComputation& preComputation) const;
    ocs2::VectorFunctionLinearApproximation getVelocityLinearApproximation(const OCPPreComputation& preComputation) const;
    ocs2::VectorFunctionLinearApproximation getOrientationErrorLinearApproximation(const OCPPreComputation& preComputation, const quaternion_t& referenceOrientation) const;
    ocs2::VectorFunctionLinearApproximation getAngularVelocityLinearApproximation(const OCPPreComputation& preComputation) const;
    ocs2::VectorFunctionLinearApproximation getTwistLinearApproximation(const OCPPreComputation& preComputation) const;
    ocs2::VectorFunctionLinearApproximation getLinearAccelerationLinearApproximation(const OCPPreComputation& preComputation) const;
    ocs2::VectorFunctionLinearApproximation getAngularAccelerationLinearApproximation(const OCPPreComputation& preComputation) const;
    ocs2::VectorFunctionLinearApproximation getAccelerationsLinearApproximation(const OCPPreComputation& preComputation) const;

  private:
    PinocchioFrameDynamics(const PinocchioFrameDynamics& rhs);

    std::string frameName_;
    size_t frameId_;
    size_t contactIndex_ = 0;
    bool useContactCandidate_ = false;

    StateConverter<ocs2::scalar_t>* stateConverter_;
  };

}
