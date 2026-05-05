#pragma once

#include <ocs2_core/integration/SensitivityIntegrator.h>

namespace ocp_solver {

  ocs2::DynamicsSensitivityDiscretizer selectDynamicsSensitivityDiscretization(ocs2::SensitivityIntegratorType integratorType);
  ocs2::VectorFunctionLinearApproximation eulerSensitivityDiscretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt);

  ocs2::VectorFunctionLinearApproximation rk2SensitivityDiscretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt);

  ocs2::VectorFunctionLinearApproximation rk4SensitivityDiscretization(ocs2::SystemDynamicsBase& system, ocs2::scalar_t t, const ocs2::vector_t& x, const ocs2::vector_t& u, ocs2::scalar_t dt);

}
