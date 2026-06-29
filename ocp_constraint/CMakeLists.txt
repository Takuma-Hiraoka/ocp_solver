cmake_minimum_required(VERSION 3.16)
project(ocp_constraint)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)

set(dependencies
  ocp_solver
)

foreach(dep ${dependencies})
  find_package(${dep} REQUIRED)
endforeach()

add_library(${PROJECT_NAME}
  src/penalties/piece_wise_polynominal_barrier_penalty.cpp
  src/position_constraint.cpp
  src/contact_fix_constraint.cpp
  src/point_constraint.cpp
  src/position_constraint_ad.cpp
  src/contact_fix_constraint_ad.cpp
  src/swing_position_constraint.cpp
  src/swing_position_constraint_ad.cpp
  src/joint_limit_constraint.cpp
  src/cop_constraint.cpp
  src/surface_contact_constraint.cpp
  src/target_wrench_constraint.cpp
  src/joint_torque_cost.cpp
  src/joint_torque_cost_ad.cpp
  )

target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
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
