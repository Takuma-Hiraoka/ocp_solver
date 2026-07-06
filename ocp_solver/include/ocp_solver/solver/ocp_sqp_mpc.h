#pragma once

#include <ocs2_mpc/MPC_BASE.h>
#include <algorithm>
#include <iostream>

#include "ocp_solver/solver/ocp_sqp_solver.h"

namespace ocp_solver {

  class OcpSqpMpc final : public ocs2::MPC_BASE {
  public:
  OcpSqpMpc(ocs2::mpc::Settings mpcSettings, ocs2::sqp::Settings settings, const ocs2::OptimalControlProblem& optimalControlProblem,
         const ocs2::Initializer& initializer)
    : ocs2::MPC_BASE(mpcSettings),
      runtimeMpcSettings_(std::move(mpcSettings)) {
      solverPtr_.reset(new OcpSqpSolver(std::move(settings), optimalControlProblem, initializer));
    };

    ~OcpSqpMpc() override = default;

    bool run(ocs2::scalar_t currentTime, const ocs2::vector_t& currentState) override {
      if (!initRun_ && currentTime >= solverPtr_->getFinalTime()) {
        std::cerr << "WARNING: The MPC time-horizon is smaller than the MPC starting time.\n";
        std::cerr << "currentTime: " << currentTime << "\t Controller finalTime: " << solverPtr_->getFinalTime() << '\n';
        return false;
      }

      calculateController(currentTime, currentState, currentTime + runtimeMpcSettings_.timeHorizon_);
      initRun_ = false;
      return true;
    }

    void reset() override {
      initRun_ = true;
      solverPtr_->reset();
    }

    OcpSqpSolver* getSolverPtr() override { return solverPtr_.get(); }
    const OcpSqpSolver* getSolverPtr() const override { return solverPtr_.get(); }

    void setTimeHorizon(ocs2::scalar_t timeHorizon) {
      runtimeMpcSettings_.timeHorizon_ = std::max<ocs2::scalar_t>(timeHorizon, 1.0e-6);
    }

    ocs2::scalar_t getRuntimeTimeHorizon() const {
      return runtimeMpcSettings_.timeHorizon_;
    }

  protected:
    void calculateController(ocs2::scalar_t initTime, const ocs2::vector_t& initState, ocs2::scalar_t finalTime) override {
      if (settings().coldStart_) {
        solverPtr_->reset();
      }
      solverPtr_->run(initTime, initState, finalTime);
    }

  private:
    std::unique_ptr<OcpSqpSolver> solverPtr_;
    ocs2::mpc::Settings runtimeMpcSettings_;
    bool initRun_ = true;
  };

}
