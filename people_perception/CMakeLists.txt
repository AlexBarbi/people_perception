cmake_minimum_required(VERSION 3.16)
project(people_perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The nodes of this package handle a dense cloud on every scan: not -O0 because no build type was given.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(autoware_perception_msgs REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(people_perception_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

###########
## Build ##
###########

# The node classes are in a library so the tests can run them in process.
add_library(${PROJECT_NAME}_nodes STATIC
  src/CloudFilter.cpp
  src/CloudLayout.cpp
  src/DetectorCloud.cpp
  src/DetectorCloudAdapter.cpp
  src/DynamicPointsFilter.cpp
  src/ObstacleAdapter.cpp
  src/PeopleTracker.cpp
  src/PeopleTrackerNode.cpp
)
target_include_directories(${PROJECT_NAME}_nodes PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(${PROJECT_NAME}_nodes PUBLIC Eigen3::Eigen)
ament_target_dependencies(${PROJECT_NAME}_nodes PUBLIC
  autoware_perception_msgs
  diagnostic_updater
  geometry_msgs
  nav_msgs
  people_perception_msgs
  rclcpp
  sensor_msgs
  tf2
  tf2_eigen
  tf2_ros
  visualization_msgs
)
target_compile_options(${PROJECT_NAME}_nodes PRIVATE -Wall -Wextra)

set(node_executables detector_cloud_adapter people_tracker dynamic_points_filter obstacle_adapter)
foreach(node ${node_executables})
  add_executable(${node} src/${node}.cpp)
  target_link_libraries(${node} ${PROJECT_NAME}_nodes)
  target_compile_options(${node} PRIVATE -Wall -Wextra)
endforeach()

##########
## Test ##
##########

if(BUILD_TESTING)
  # No ROS graph in these
  find_package(ament_cmake_gtest REQUIRED)
  foreach(test_name test_people_tracker test_cloud_ops)
    ament_add_gtest(${test_name} test/${test_name}.cpp)
    target_link_libraries(${test_name} ${PROJECT_NAME}_nodes)
  endforeach()

  # ROS-isolated: the test nodes get a ROS_DOMAIN_ID of their own, away from a running stack.
  find_package(ament_cmake_ros REQUIRED)
  find_package(diagnostic_msgs REQUIRED)
  find_package(tf2_msgs REQUIRED)
  foreach(test_name test_detector_cloud_adapter test_people_tracker_node test_dynamic_points_filter
          test_obstacle_adapter)
    ament_add_ros_isolated_gtest(${test_name} test/${test_name}.cpp TIMEOUT 120)
    target_link_libraries(${test_name} ${PROJECT_NAME}_nodes)
    ament_target_dependencies(${test_name} diagnostic_msgs tf2_msgs)
    target_compile_definitions(${test_name} PRIVATE
      PEOPLE_PERCEPTION_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config"
    )
  endforeach()

  # The dataset export, on bags written by the test
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_export_detector_dataset test/test_export_detector_dataset.py)
  # The people as the obstacles of the Opti-Pessi controller
  ament_add_pytest_test(test_dynamic_obstacles_to_opti_pessi_obstacles
    test/test_dynamic_obstacles_to_opti_pessi_obstacles.py)
endif()

#############
## Install ##
#############

install(TARGETS ${node_executables}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS scripts/export_detector_dataset.py
                 scripts/dynamic_obstacles_to_opti_pessi_obstacles.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
