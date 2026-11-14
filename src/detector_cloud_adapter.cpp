// Runs people_perception::DetectorCloudAdapter (see include/people_perception/DetectorCloudAdapter.h).
//
// The parameters have no defaults in the code: people_perception_launch.xml people:=true starts it with its yaml.

#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "people_perception/DetectorCloudAdapter.h"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    rclcpp::spin(std::make_shared<people_perception::DetectorCloudAdapter>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("detector_cloud_adapter"), "%s", e.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
