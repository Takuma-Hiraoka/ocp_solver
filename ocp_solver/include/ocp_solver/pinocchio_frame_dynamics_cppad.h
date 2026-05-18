#pragma once

#include <functional>
#include <string>
#include <vector>

#include <ocs2_core/Types.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include <ocs2_core/automatic_differentiation/CppAdInterface.h>

#include "ocp_solver/state_converter.h"

namespace ocp_solver {

  class PinocchioFrameDynamicsCppAd final : public ocs2::EndEffectorKinematics<ocs2::scalar_t> {
  public:
    using vector3_t = Eigen::Matrix<ocs2::scalar_t, 3, 1>;
    using matrix3x_t = Eigen::Matrix<ocs2::scalar_t, 3, Eigen::Dynamic>;
    using vector6_t = Eigen::Matrix<ocs2::scalar_t, 6, 1>;
    using matrix6x_t = Eigen::Matrix<ocs2::scalar_t, 6, Eigen::Dynamic>;
    using vector_t = Eigen::Matrix<ocs2::scalar_t, Eigen::Dynamic, 1>;
    using quaternion_t = Eigen::Quaternion<ocs2::scalar_t>;
    using update_pinocchio_interface_callback =
      std::function<void(const ocs2::ad_vector_t& state, ocs2::PinocchioInterfaceTpl<ocs2::ad_scalar_t>& pinocchioInterface)>;

    PinocchioFrameDynamicsCppAd(const ocs2::PinocchioInterface& pinocchioInterface,
                                      StateConverter<ocs2::ad_scalar_t>& stateConverter,
                                      std::vector<std::string> frameNames,
                                      const std::string& modelName,
                                      const std::string& modelFolder = "build/cppad_autocode_gen",
                                      bool recompileLibraries = false,
                                      bool verbose = false);

    ~PinocchioFrameDynamicsCppAd() override = default;
    PinocchioFrameDynamicsCppAd* clone() const override;
    PinocchioFrameDynamicsCppAd& operator=(const PinocchioFrameDynamicsCppAd&) = delete;

    const std::vector<std::string>& getIds() const override { return frameNames_; }
    const std::vector<std::size_t>& getFrameIds() const { return frameIds_; }

    std::vector<vector3_t> getPosition(const vector_t& state) const override;
    std::vector<vector3_t> getVelocity(const vector_t& state, const vector_t& input) const override;

    std::vector<quaternion_t> getOrientation(const vector_t& state) const;
    std::vector<vector3_t> getOrientationError(const vector_t& state, const std::vector<quaternion_t>& referenceOrientations) const override;
    std::vector<vector3_t> getOrientationErrorWrtPlane(const vector_t& state, const std::vector<vector3_t>& planeNormals) const;
    std::vector<vector3_t> getAngularVelocity(const vector_t& state, const vector_t& input) const;
    std::vector<vector6_t> getTwist(const vector_t& state, const vector_t& input) const;
    std::vector<vector3_t> getLinearAcceleration(const vector_t& state, const vector_t& input) const;
    std::vector<vector3_t> getAngularAcceleration(const vector_t& state, const vector_t& input) const;
    std::vector<vector6_t> getAccelerations(const vector_t& state, const vector_t& input) const;

    std::vector<ocs2::VectorFunctionLinearApproximation> getPositionLinearApproximation(const vector_t& state) const override;
    std::vector<ocs2::VectorFunctionLinearApproximation> getVelocityLinearApproximation(const vector_t& state,
                                                                                        const vector_t& input) const override;
    std::vector<ocs2::VectorFunctionLinearApproximation> getOrientationErrorLinearApproximation(
                                                                                                const vector_t& state, const std::vector<quaternion_t>& referenceOrientations) const override;
    std::vector<ocs2::VectorFunctionLinearApproximation> getAngularVelocityLinearApproximation(const vector_t& state,
                                                                                               const vector_t& input) const;
    std::vector<ocs2::VectorFunctionLinearApproximation> getTwistLinearApproximation(const vector_t& state, const vector_t& input) const;
    std::vector<ocs2::VectorFunctionLinearApproximation> getLinearAccelerationLinearApproximation(const vector_t& state,
                                                                                                  const vector_t& input) const;
    std::vector<ocs2::VectorFunctionLinearApproximation> getAngularAccelerationLinearApproximation(const vector_t& state,
                                                                                                   const vector_t& input) const;
    std::vector<ocs2::VectorFunctionLinearApproximation> getAccelerationsLinearApproximation(const vector_t& state,
                                                                                             const vector_t& input) const;

    ocs2::ad_vector_t getPositionCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& p);
    ocs2::ad_vector_t getVelocityCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p);
    ocs2::ad_vector_t getOrientationCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& p);
    ocs2::ad_vector_t getOrientationErrorCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& params);
    ocs2::ad_vector_t getAngularVelocityCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p);
    ocs2::ad_vector_t getTwistCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p);
    ocs2::ad_vector_t getLinearAccelerationCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p);
    ocs2::ad_vector_t getAngularAccelerationCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p);
    ocs2::ad_vector_t getAccelerationsCppAd(const ocs2::ad_vector_t& dx, const ocs2::ad_vector_t& input, const ocs2::ad_vector_t& p);

  private:
    PinocchioFrameDynamicsCppAd(const PinocchioFrameDynamicsCppAd& rhs);

    std::unique_ptr<ocs2::CppAdInterface> positionCppAdInterfacePtr_;
    std::unique_ptr<ocs2::CppAdInterface> velocityCppAdInterfacePtr_;
    std::unique_ptr<ocs2::CppAdInterface> orientationCppAdInterfacePtr_;
    std::unique_ptr<ocs2::CppAdInterface> orientationErrorCppAdInterfacePtr_;
    std::unique_ptr<ocs2::CppAdInterface> angularVelocityCppAdInterfacePtr_;
    std::unique_ptr<ocs2::CppAdInterface> twistCppAdInterfacePtr_;
    std::unique_ptr<ocs2::CppAdInterface> linearAccelerationCppAdInterfacePtr_;
    std::unique_ptr<ocs2::CppAdInterface> angularAccelerationCppAdInterfacePtr_;
    std::unique_ptr<ocs2::CppAdInterface> accelerationsCppAdInterfacePtr_;

    const std::vector<std::string> frameNames_;
    std::vector<size_t> frameIds_;

    ocs2::PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
    StateConverter<ocs2::ad_scalar_t>* mappingPtr_;
  };

}
