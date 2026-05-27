#include <unordered_map>
#include "ocp_solver/solver/ocp_sensitivity_integrator.h"
#include "ocp_solver/solver/system_dynamics_ad.h"
#include "ocp_solver/solver/system_dynamics.h"

namespace ocp_solver {
  ocs2::DynamicsDiscretizer selectDynamicsDiscretization(ocs2::SensitivityIntegratorType integratorType) {
    switch (integratorType) {
    case ocs2::SensitivityIntegratorType::EULER:
      return eulerDiscretization;
    case ocs2::SensitivityIntegratorType::RK2:
      return rk2Discretization;
    case ocs2::SensitivityIntegratorType::RK4:
      return rk4Discretization;
    default:
      throw std::runtime_error("Integrator of type not supported.");
    }
  }

  ocs2::vector_t eulerDiscretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt) {
    SystemDynamicsAD* systemADPtr = dynamic_cast<SystemDynamicsAD*>(&system);
    SystemDynamics* systemPtr = dynamic_cast<SystemDynamics*>(&system);
    ocs2::PinocchioInterface& pinocchioInterface = systemADPtr ? systemADPtr->getPinocchioInterface() : systemPtr->getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();

    ocs2::vector_t dx = system.computeFlowMap(t, x, u);
    ocs2::vector_t tmp = x;
    tmp.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt * dx.head(model.nv));
    tmp.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt * dx.segment(model.nv, model.nv);
    return tmp;
  }

  ocs2::vector_t rk2Discretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt) {
    const ocs2::scalar_t dt_halve = dt / 2.0;
    SystemDynamicsAD* systemADPtr = dynamic_cast<SystemDynamicsAD*>(&system);
    SystemDynamics* systemPtr = dynamic_cast<SystemDynamics*>(&system);
    ocs2::PinocchioInterface& pinocchioInterface = systemADPtr ? systemADPtr->getPinocchioInterface() : systemPtr->getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();

    // System evaluations
    ocs2::vector_t k1 = system.computeFlowMap(t, x, u);
    ocs2::vector_t tmpV = x;
    tmpV.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt * k1.head(model.nv));
    tmpV.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt * k1.segment(model.nv, model.nv);
    ocs2::vector_t k2 = system.computeFlowMap(t + dt, tmpV, u);

    tmpV = x;
    tmpV.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt_halve * k1.head(model.nv));
    tmpV.head(model.nq) = pinocchio::integrate(model, tmpV.head(model.nq), dt_halve * k2.head(model.nv));
    tmpV.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt_halve * k1.segment(model.nv, model.nv);
    tmpV.segment(model.nq, model.nv) += dt_halve * k2.segment(model.nv, model.nv);

    return tmpV;
  }

  ocs2::vector_t rk4Discretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt) {
    const ocs2::scalar_t dt_halve = dt / 2.0;
    const ocs2::scalar_t dt_sixth = dt / 6.0;
    const ocs2::scalar_t dt_third = dt / 3.0;

    SystemDynamicsAD* systemADPtr = dynamic_cast<SystemDynamicsAD*>(&system);
    SystemDynamics* systemPtr = dynamic_cast<SystemDynamics*>(&system);
    ocs2::PinocchioInterface& pinocchioInterface = systemADPtr ? systemADPtr->getPinocchioInterface() : systemPtr->getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    // System evaluations
    const ocs2::vector_t k1 = system.computeFlowMap(t, x, u);
    ocs2::vector_t tmp = x;
    tmp.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt_halve * k1.head(model.nv));
    tmp.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt_halve * k1.segment(model.nv, model.nv);
    const ocs2::vector_t k2 = system.computeFlowMap(t + dt_halve, tmp, u);
    tmp.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt_halve * k2.head(model.nv));
    tmp.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt_halve * k2.segment(model.nv, model.nv);
    const ocs2::vector_t k3 = system.computeFlowMap(t + dt_halve, tmp, u);
    tmp.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt * k3.head(model.nv));
    tmp.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt * k3.segment(model.nv, model.nv);
    const ocs2::vector_t k4 = system.computeFlowMap(t + dt, tmp, u);

    tmp = x;
    tmp.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt_sixth * k1.head(model.nv));
    tmp.head(model.nq) = pinocchio::integrate(model, tmp.head(model.nq), dt_third * k2.head(model.nv));
    tmp.head(model.nq) = pinocchio::integrate(model, tmp.head(model.nq), dt_third * k3.head(model.nv));
    tmp.head(model.nq) = pinocchio::integrate(model, tmp.head(model.nq), dt_sixth * k4.head(model.nv));
    tmp.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt_sixth * k1.segment(model.nv, model.nv);
    tmp.segment(model.nq, model.nv) += dt_third * k2.segment(model.nv, model.nv);
    tmp.segment(model.nq, model.nv) += dt_third * k3.segment(model.nv, model.nv);
    tmp.segment(model.nq, model.nv) += dt_sixth * k4.segment(model.nv, model.nv);

    return tmp;
  }


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
    SystemDynamicsAD* systemADPtr = dynamic_cast<SystemDynamicsAD*>(&system);
    SystemDynamics* systemPtr = dynamic_cast<SystemDynamics*>(&system);
    ocs2::PinocchioInterface& pinocchioInterface = systemADPtr ? systemADPtr->getPinocchioInterface() : systemPtr->getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();

    ocs2::VectorFunctionLinearApproximation continuousApproximation = system.linearApproximation(t, x, u);
    continuousApproximation.dfdx *= dt;
    continuousApproximation.dfdx.diagonal().array() += 1.0;  // plus Identity()
    continuousApproximation.dfdu *= dt;
    ocs2::vector_t df = continuousApproximation.f;
    continuousApproximation.f = x;
    continuousApproximation.f.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt * df.head(model.nv));
    continuousApproximation.f.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt * df.segment(model.nv, model.nv);

    return continuousApproximation;
  }

  ocs2::VectorFunctionLinearApproximation rk2SensitivityDiscretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt) {
    const ocs2::scalar_t dt_halve = dt / 2.0;
    SystemDynamicsAD* systemADPtr = dynamic_cast<SystemDynamicsAD*>(&system);
    SystemDynamics* systemPtr = dynamic_cast<SystemDynamics*>(&system);
    ocs2::PinocchioInterface& pinocchioInterface = systemADPtr ? systemADPtr->getPinocchioInterface() : systemPtr->getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();

    // System evaluations
    ocs2::VectorFunctionLinearApproximation k1 = system.linearApproximation(t, x, u);
    ocs2::vector_t tmpV = x;
    tmpV.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt * k1.f.head(model.nv));
    tmpV.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt * k1.f.segment(model.nv, model.nv);
    ocs2::VectorFunctionLinearApproximation k2 = system.linearApproximation(t + dt, tmpV, u);

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
    k1.f = x;
    k1.f.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt_halve * k1df.head(model.nv));
    k1.f.head(model.nq) = pinocchio::integrate(model, k1.f.head(model.nq), dt_halve * k2df.head(model.nv));
    k1.f.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt_halve * k1df.segment(model.nv, model.nv);
    k1.f.segment(model.nq, model.nv) += dt_halve * k2df.segment(model.nv, model.nv);

    return k1;
  }

  ocs2::VectorFunctionLinearApproximation rk4SensitivityDiscretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt) {
    const ocs2::scalar_t dt_halve = dt / 2.0;
    const ocs2::scalar_t dt_sixth = dt / 6.0;
    const ocs2::scalar_t dt_third = dt / 3.0;

    SystemDynamicsAD* systemADPtr = dynamic_cast<SystemDynamicsAD*>(&system);
    SystemDynamics* systemPtr = dynamic_cast<SystemDynamics*>(&system);
    ocs2::PinocchioInterface& pinocchioInterface = systemADPtr ? systemADPtr->getPinocchioInterface() : systemPtr->getPinocchioInterface();
    const pinocchio::Model& model = pinocchioInterface.getModel();
    // System evaluations
    ocs2::VectorFunctionLinearApproximation k1 = system.linearApproximation(t, x, u);
    ocs2::vector_t tmpV = x;
    tmpV.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt_halve * k1.f.head(model.nv));
    tmpV.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt_halve * k1.f.segment(model.nv, model.nv);
    ocs2::VectorFunctionLinearApproximation k2 = system.linearApproximation(t + dt_halve, tmpV, u);
    tmpV.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt_halve * k2.f.head(model.nv));
    tmpV.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt_halve * k2.f.segment(model.nv, model.nv);
    ocs2::VectorFunctionLinearApproximation k3 = system.linearApproximation(t + dt_halve, tmpV, u);
    tmpV.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt * k3.f.head(model.nv));
    tmpV.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt * k3.f.segment(model.nv, model.nv);
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
    k1.f = x;
    k1.f.head(model.nq) = pinocchio::integrate(model, x.head(model.nq), dt_sixth * k1df.head(model.nv));
    k1.f.head(model.nq) = pinocchio::integrate(model, k1.f.head(model.nq), dt_third * k2.f.head(model.nv));
    k1.f.head(model.nq) = pinocchio::integrate(model, k1.f.head(model.nq), dt_third * k3.f.head(model.nv));
    k1.f.head(model.nq) = pinocchio::integrate(model, k1.f.head(model.nq), dt_sixth * k4.f.head(model.nv));
    k1.f.segment(model.nq, model.nv) = x.segment(model.nq, model.nv) + dt_sixth * k1df.segment(model.nv, model.nv);
    k1.f.segment(model.nq, model.nv) += dt_third * k2.f.segment(model.nv, model.nv);
    k1.f.segment(model.nq, model.nv) += dt_third * k3.f.segment(model.nv, model.nv);
    k1.f.segment(model.nq, model.nv) += dt_sixth * k4.f.segment(model.nv, model.nv);

    return k1;
  }

}
