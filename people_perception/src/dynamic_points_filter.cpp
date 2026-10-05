// Runs people_perception::DynamicPointsFilter (see include/people_perception/DynamicPointsFilter.h).
//
// fast_lio fast_lio_slam_launch.xml starts it (through people_perception_launch.xml): do not start a second one next
// to it. The parameters have no defaults in the code.

#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "people_perception/DynamicPointsFilter.h"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    rclcpp::spin(std::make_shared<people_perception::DynamicPointsFilter>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("dynamic_points_filter"), "%s", e.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
