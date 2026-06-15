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
  osqp_eigen
)

foreach(dep ${dependencies})
  find_package(${dep} REQUIRED)
endforeach()

message(WARNING "pinocchiodir ${pinocchio_DIR}")
add_library(${PROJECT_NAME}
  src/ocp_interface.cpp
  src/pose_optimizer.cpp
  src/contact_schedule.cpp
  src/solver/switched_model_reference_manager.cpp
  src/solver/trajectory.cpp
  src/solver/dynamics_helper_functions.cpp
  src/solver/system_dynamics.cpp
  src/solver/system_dynamics_ad.cpp
  src/solver/ocp_pre_computation.cpp
  src/solver/gravity_compensation_initializer.cpp
  src/solver/ocp_sqp_solver.cpp
  src/solver/ocp_transcription.cpp
  src/solver/ocp_sensitivity_integrator.cpp
  src/solver/ocp_metrics_computation.cpp
  src/pinocchio/pinocchio_frame_dynamics_cppad.cpp
  src/pinocchio/pinocchio_frame_dynamics.cpp
  src/common/zero_wrench_constraint.cpp
  src/common/quadratic_state_cost.cpp
  src/common/quadratic_state_input_cost.cpp
  src/ocp_data/ocp_optimal_control_problem.cpp
  src/ocp_data/ocp_state_cost_collection.cpp
  src/ocp_data/ocp_state_input_cost_collection.cpp
  src/ocp_data/ocp_state_constraint_collection.cpp
  src/ocp_data/ocp_state_input_constraint_collection.cpp
  )

target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Boost_INCLUDE_DIRS}
  )

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  Boost::system
  Boost::filesystem
  Boost::log
  Boost::log_setup
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

ament_package(CONFIG_EXTRAS "cmake/ocp_solver-extras.cmake")
