cmake_minimum_required(VERSION 3.16)
project(ocp_solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS
  system
  filesystem
  log
  log_setup
  )


set(dependencies
  Eigen3
  ocs2_robotic_tools
  ocs2_core
  ocs2_pinocchio_interface
  ocs2_oc
  ocs2_mpc
  ocs2_sqp
  pinocchio_model_builder
)

foreach(dep ${dependencies})
  find_package(${dep} REQUIRED)
endforeach()

message(WARNING "pinocchiodir ${pinocchio_DIR}")
add_library(${PROJECT_NAME}
  src/switched_model_reference_manager.cpp
  src/contact_schedule.cpp
  src/ocp_interface.cpp
  src/dynamics_helper_functions.cpp
  src/dynamics_helper_functions_ad.cpp
  src/system_dynamics.cpp
  src/system_dynamics_ad.cpp
  src/ocp_pre_computation.cpp
  src/gravity_compensation_initializer.cpp
  src/ocp_pinocchio_mapping.cpp
  src/pinocchio_endeffector_dynamics_cppad.cpp
  src/zero_wrench_constraint.cpp
  src/ocp_sqp_solver.cpp
  src/ocp_transcription.cpp
  src/ocp_sensitivity_integrator.cpp
  src/ocp_metrics_computation.cpp
  src/quadratic_state_cost.cpp
  src/quadratic_state_input_cost.cpp
  )

target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Boost_INCLUDE_DIRS}
  )

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  )

ament_target_dependencies(${PROJECT_NAME}
  ${dependencies}
  )

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )

install(DIRECTORY include/
  DESTINATION include
  )

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${dependencies})

ament_package()
