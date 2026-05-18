#pragma once

#include <memory>

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocp_solver/pinocchio/pinocchio_frame_dynamics_cppad.h>
#include <ocp_solver/solver/switched_model_reference_manager.h>

namespace ocp_constraint {

  class SwingPositionConstraintAD final : public ocs2::StateConstraint {
  public:
    struct Config {
      Config(){};
      ocs2::matrix_t Ax = Eigen::MatrixXd::Identity(3, 3);
    };

    SwingPositionConstraintAD(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                              const ocp_solver::PinocchioFrameDynamicsCppAd& endFrameDynamics,
                              Config config = Config(),
                              ocs2::scalar_t ignoreTime = 0.5,
                              double height = 0.1,
                              double swingWeight = 3.0);

    ~SwingPositionConstraintAD() override = default;
    SwingPositionConstraintAD* clone() const override { return new SwingPositionConstraintAD(*this); }

    void configure(Config&& config);
    void configure(const Config& config) { this->configure(Config(config)); }

    ocp_solver::PinocchioFrameDynamicsCppAd& getFrameDynamics() { return *frameDynamicsPtr_; }

    bool isActive(ocs2::scalar_t time) const override;
    size_t getNumConstraints(ocs2::scalar_t time) const override { return 3; }
    std::vector<std::pair<ocs2::scalar_t, pinocchio::SE3> > nearestContact(ocs2::scalar_t time) const;
    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::PreComputation& preComp) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::PreComputation& preComp) const override;

  private:
    SwingPositionConstraintAD(const SwingPositionConstraintAD& rhs);
    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_;
    std::unique_ptr<ocp_solver::PinocchioFrameDynamicsCppAd> frameDynamicsPtr_;
    Config config_;
    ocs2::scalar_t ignoreTime_ = 0.5;
    double height_ = 0.1;
    double swingWeight_ = 3.0;
  };

}
