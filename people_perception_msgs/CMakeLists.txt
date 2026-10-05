cmake_minimum_required(VERSION 3.16)
project(people_perception_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unique_identifier_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/DynamicObstacle.msg
  msg/DynamicObstacleArray.msg
  DEPENDENCIES geometry_msgs std_msgs unique_identifier_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
