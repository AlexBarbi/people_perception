// Runs people_perception::PeopleTrackerNode (see include/people_perception/PeopleTrackerNode.h).
//
// The parameters have no defaults in the code: people_perception_launch.xml people:=true starts it with its yaml.

#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "people_perception/PeopleTrackerNode.h"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    rclcpp::spin(std::make_shared<people_perception::PeopleTrackerNode>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("people_tracker"), "%s", e.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
