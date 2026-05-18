#pragma once

#include <memory>

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocp_solver/pinocchio_frame_dynamics.h>
#include <ocp_solver/switched_model_reference_manager.h>

namespace ocp_constraint {

  class SwingPositionConstraint final : public ocs2::StateConstraint {
  public:
    struct Config {
      Config(){};
      ocs2::matrix_t Ax = Eigen::MatrixXd::Identity(3, 3);
    };

    SwingPositionConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                            const ocp_solver::PinocchioFrameDynamics& frameDynamics,
                            Config config = Config(),
                            ocs2::scalar_t ignoreTime = 0.5,
                            double height = 0.1,
                            double swingWeight = 3.0);

    ~SwingPositionConstraint() override = default;
    SwingPositionConstraint* clone() const override { return new SwingPositionConstraint(*this); }

    void configure(Config&& config);
    void configure(const Config& config) { this->configure(Config(config)); }

    ocp_solver::PinocchioFrameDynamics& getFrameDynamics() { return *frameDynamicsPtr_; }

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
    SwingPositionConstraint(const SwingPositionConstraint& rhs);
    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_;
    std::unique_ptr<ocp_solver::PinocchioFrameDynamics> frameDynamicsPtr_;
    Config config_;
    ocs2::scalar_t ignoreTime_ = 0.5;
    double height_ = 0.1;
    double swingWeight_ = 3.0;
  };

}
