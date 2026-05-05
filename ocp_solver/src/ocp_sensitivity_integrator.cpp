#include <unordered_map>
#include "ocp_solver/ocp_sensitivity_integrator.h"
#include "ocp_solver/system_dynamics_ad.h"

namespace ocp_solver {
  ocs2::DynamicsSensitivityDiscretizer selectDynamicsSensitivityDiscretization(ocs2::SensitivityIntegratorType integratorType) {
    switch (integratorType) {
    case ocs2::SensitivityIntegratorType::EULER:
      return eulerSensitivityDiscretization;
    case ocs2::SensitivityIntegratorType::RK2:
      return rk2SensitivityDiscretization;
    case ocs2::SensitivityIntegratorType::RK4:
      return rk4SensitivityDiscretization;
    default:
      throw std::runtime_error("Integrator of type not supported.");
    }
  }

  ocs2::VectorFunctionLinearApproximation eulerSensitivityDiscretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt) {
    // x_{k+1} = A_{k} * dx_{k} + B_{k} * du_{k} + b_{k}
    // A_{k} = Id + dt * dfdx
    // B_{k} = dt * dfdu
    // b_{k} = x_{n} + dt * f(x_{n},u_{n})
    ocs2::PinocchioInterface& pinocchioInterface = static_cast<SystemDynamicsAD&>(system).getPinocchioInterface();

    auto continuousApproximation = system.linearApproximation(t, x, u);
    continuousApproximation.dfdx *= dt;
    continuousApproximation.dfdx.diagonal().array() += 1.0;  // plus Identity()
    continuousApproximation.dfdu *= dt;
    ocs2::vector_t df = continuousApproximation.f;
    continuousApproximation.f = ocs2::vector_t::Zero(x.size());
    continuousApproximation.f.head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), x.head(pinocchioInterface.getModel().nq), dt * df.head(pinocchioInterface.getModel().nv));
    continuousApproximation.f.tail(pinocchioInterface.getModel().nv) = x.tail(pinocchioInterface.getModel().nv) + dt * df.tail(pinocchioInterface.getModel().nv);

    return continuousApproximation;
  }

  ocs2::VectorFunctionLinearApproximation rk2SensitivityDiscretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt) {
    const ocs2::scalar_t dt_halve = dt / 2.0;
    ocs2::PinocchioInterface& pinocchioInterface = static_cast<SystemDynamicsAD&>(system).getPinocchioInterface();

    // System evaluations
    ocs2::VectorFunctionLinearApproximation k1 = system.linearApproximation(t, x, u);
    ocs2::VectorFunctionLinearApproximation k2 = system.linearApproximation(t + dt, x + dt * k1.f, u);

    // Input sensitivity \dot{Su} = dfdx(t) Su + dfdu(t), with Su(0) = Zero()
    // Re-use memory from k.dfdu as dkduk
    // dk1duk = k1.dfdu
    k2.dfdu.noalias() += dt * k2.dfdx * k1.dfdu;

    // State sensitivity \dot{Sx} = dfdx(t) Sx, with Sx(0) = Identity()
    // Re-use memory from k.dfdx as dkdxk
    // dk1dxk = k1.dfdx;
    k2.dfdx += dt * k2.dfdx * k1.dfdx;  // need one temporary to avoid alias

    // Assemble discrete approximation
    // Re-use k1 to collect the result
    k1.dfdx = dt_halve * k1.dfdx + dt_halve * k2.dfdx;
    k1.dfdx.diagonal().array() += 1.0;  // plus Identity()
    k1.dfdu = dt_halve * k1.dfdu + dt_halve * k2.dfdu;

    ocs2::vector_t k1df = k1.f;
    ocs2::vector_t k2df = k2.f;
    k1.f = ocs2::vector_t::Zero(x.size());
    k1.f.head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), x.head(pinocchioInterface.getModel().nq), dt_halve * k1df.head(pinocchioInterface.getModel().nv));
    k1.f.head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), k1.f.head(pinocchioInterface.getModel().nq), dt_halve * k2df.head(pinocchioInterface.getModel().nv));
    k1.f.tail(pinocchioInterface.getModel().nv) = x.tail(pinocchioInterface.getModel().nv) + dt_halve * k1df.tail(pinocchioInterface.getModel().nv);
    k1.f.tail(pinocchioInterface.getModel().nv) += dt_halve * k2df.tail(pinocchioInterface.getModel().nv);

    return k1;
  }

  ocs2::VectorFunctionLinearApproximation rk4SensitivityDiscretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt) {
    const ocs2::scalar_t dt_halve = dt / 2.0;
    const ocs2::scalar_t dt_sixth = dt / 6.0;
    const ocs2::scalar_t dt_third = dt / 3.0;

    ocs2::PinocchioInterface& pinocchioInterface = static_cast<SystemDynamicsAD&>(system).getPinocchioInterface();
    // System evaluations
    ocs2::VectorFunctionLinearApproximation k1 = system.linearApproximation(t, x, u);
    ocs2::vector_t tmpV = x;
    tmpV.head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), x.head(pinocchioInterface.getModel().nq), dt_halve * k1.f.head(pinocchioInterface.getModel().nv));
    tmpV.tail(pinocchioInterface.getModel().nv) = x.tail(pinocchioInterface.getModel().nv) + dt * k1.f.tail(pinocchioInterface.getModel().nv);
    ocs2::VectorFunctionLinearApproximation k2 = system.linearApproximation(t + dt_halve, tmpV, u);
    tmpV.head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), x.head(pinocchioInterface.getModel().nq), dt_halve * k2.f.head(pinocchioInterface.getModel().nv));
    tmpV.tail(pinocchioInterface.getModel().nv) = x.tail(pinocchioInterface.getModel().nv) + dt * k2.f.tail(pinocchioInterface.getModel().nv);
    ocs2::VectorFunctionLinearApproximation k3 = system.linearApproximation(t + dt_halve, tmpV, u);
    tmpV.head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), x.head(pinocchioInterface.getModel().nq), dt * k3.f.head(pinocchioInterface.getModel().nv));
    tmpV.tail(pinocchioInterface.getModel().nv) = x.tail(pinocchioInterface.getModel().nv) + dt * k3.f.tail(pinocchioInterface.getModel().nv);
    ocs2::VectorFunctionLinearApproximation k4 = system.linearApproximation(t + dt, tmpV, u);

    // Input sensitivity \dot{Su} = dfdx(t) Su + dfdu(t), with Su(0) = Zero()
    // Re-use memory from k.dfdu as dkduk
    // dk1duk = k1.dfdu
    k2.dfdu.noalias() += dt_halve * k2.dfdx * k1.dfdu;
    k3.dfdu.noalias() += dt_halve * k3.dfdx * k2.dfdu;
    k4.dfdu.noalias() += dt * k4.dfdx * k3.dfdu;

    // State sensitivity \dot{Sx} = dfdx(t) Sx, with Sx(0) = Identity()
    // Re-use memory from k.dfdx as dkdxk
    // dk1dxk = k1.dfdx;
    ocs2::matrix_t tmp = dt_halve * k2.dfdx * k1.dfdx;  // need one temporary to avoid alias
    k2.dfdx += tmp;
    tmp.noalias() = dt_halve * k3.dfdx * k2.dfdx;
    k3.dfdx += tmp;
    tmp.noalias() = dt * k4.dfdx * k3.dfdx;
    k4.dfdx += tmp;

    // Assemble discrete approximation
    // Re-use k1 to collect the result
    k1.dfdx = dt_sixth * k1.dfdx + dt_third * k2.dfdx + dt_third * k3.dfdx + dt_sixth * k4.dfdx;
    k1.dfdx.diagonal().array() += 1.0;  // plus Identity()
    k1.dfdu = dt_sixth * k1.dfdu + dt_third * k2.dfdu + dt_third * k3.dfdu + dt_sixth * k4.dfdu;

    ocs2::vector_t k1df = k1.f;
    k1.f = ocs2::vector_t::Zero(x.size());
    k1.f.head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), x.head(pinocchioInterface.getModel().nq), dt_sixth * k1df.head(pinocchioInterface.getModel().nv));
    k1.f.head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), k1.f.head(pinocchioInterface.getModel().nq), dt_third * k2.f.head(pinocchioInterface.getModel().nv));
    k1.f.head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), k1.f.head(pinocchioInterface.getModel().nq), dt_third * k3.f.head(pinocchioInterface.getModel().nv));
    k1.f.head(pinocchioInterface.getModel().nq) = pinocchio::integrate(pinocchioInterface.getModel(), k1.f.head(pinocchioInterface.getModel().nq), dt_sixth * k4.f.head(pinocchioInterface.getModel().nv));
    k1.f.tail(pinocchioInterface.getModel().nv) = x.tail(pinocchioInterface.getModel().nv) + dt_sixth * k1df.tail(pinocchioInterface.getModel().nv);
    k1.f.tail(pinocchioInterface.getModel().nv) += dt_third * k2.f.tail(pinocchioInterface.getModel().nv);
    k1.f.tail(pinocchioInterface.getModel().nv) += dt_third * k3.f.tail(pinocchioInterface.getModel().nv);
    k1.f.tail(pinocchioInterface.getModel().nv) += dt_sixth * k4.f.tail(pinocchioInterface.getModel().nv);

    return k1;
  }

}
