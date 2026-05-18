#pragma once

#include <ocs2_core/cost/StateInputGaussNewtonCostAd.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "ocp_solver/solver/state_converter.h"

namespace ocp_constraint {

  class JointTorqueCostCppAd final : public ocs2::StateInputCostGaussNewtonAd {
  public:
    JointTorqueCostCppAd(const ocs2::vector_t& weights,
                         const ocs2::PinocchioInterface& pinocchioInterface,
                         const ocp_solver::StateConverter<ocs2::ad_scalar_t>& stateConverter,
                         std::string costName,
                         std::string modelFolder = "build/cppad_autocode_gen",
                         bool recompileLibrariesCppAd = false);

    ~JointTorqueCostCppAd() override = default;
    JointTorqueCostCppAd* clone() const override { return new JointTorqueCostCppAd(*this); }

    ocs2::vector_t getParameters(ocs2::scalar_t time,
                                 const ocs2::TargetTrajectories& targetTrajectories,
                                 const ocs2::PreComputation& preComputation) const override {
      return sqrtWeights_;
    }

  private:
    JointTorqueCostCppAd(const JointTorqueCostCppAd& other);

    ocs2::ad_vector_t costVectorFunction(ocs2::ad_scalar_t time,
                                         const ocs2::ad_vector_t& state,
                                         const ocs2::ad_vector_t& input,
                                         const ocs2::ad_vector_t& parameters) const override;

    ocs2::vector_t sqrtWeights_;

    mutable ocs2::PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
    ocp_solver::StateConverter<ocs2::ad_scalar_t>* stateConverterPtr_;
  };

}
