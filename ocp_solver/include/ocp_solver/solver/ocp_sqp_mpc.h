#pragma once

#include <ocs2_mpc/MPC_BASE.h>

#include "ocp_solver/solver/ocp_sqp_solver.h"

namespace ocp_solver {

  class OcpSqpMpc final : public ocs2::MPC_BASE {
  public:
  OcpSqpMpc(ocs2::mpc::Settings mpcSettings, ocs2::sqp::Settings settings, const ocs2::OptimalControlProblem& optimalControlProblem,
         const ocs2::Initializer& initializer)
    : ocs2::MPC_BASE(std::move(mpcSettings)) {
      solverPtr_.reset(new OcpSqpSolver(std::move(settings), optimalControlProblem, initializer));
    };

    ~OcpSqpMpc() override = default;

    OcpSqpSolver* getSolverPtr() override { return solverPtr_.get(); }
    const OcpSqpSolver* getSolverPtr() const override { return solverPtr_.get(); }

  protected:
    void calculateController(ocs2::scalar_t initTime, const ocs2::vector_t& initState, ocs2::scalar_t finalTime) override {
      if (settings().coldStart_) {
        solverPtr_->reset();
      }
      solverPtr_->run(initTime, initState, finalTime);
    }

  private:
    std::unique_ptr<OcpSqpSolver> solverPtr_;
  };

}
