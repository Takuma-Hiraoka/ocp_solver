cmake_minimum_required(VERSION 3.16)
project(ocp_constraint)

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
  ocs2_core
  ocp_solver
  ocs2_pinocchio_interface
  ocs2_robotic_tools
)

foreach(dep ${dependencies})
  find_package(${dep} REQUIRED)
endforeach()

add_library(${PROJECT_NAME}
  src/penalties/piece_wise_polynominal_barrier_penalty.cpp
  src/position_constraint.cpp
  src/point_constraint.cpp
  src/position_constraint_ad.cpp
  src/swing_position_constraint_ad.cpp
  src/joint_limit_constraint.cpp
  src/surface_contact_constraint.cpp
  src/joint_torque_cost_ad.cpp
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
