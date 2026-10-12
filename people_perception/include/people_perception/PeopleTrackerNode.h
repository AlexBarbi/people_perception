#pragma once

// ROS 2 node: block B5 of docs/perception_pipeline.md, the PeopleTracker of PeopleTracker.h on the detections of
// autoware_lidar_centerpoint. For every DetectedObjects message it brings the pedestrians to odom_frame with the TF at
// the stamp of the message, updates the tracks and publishes, with that stamp and in odom_frame:
//   tracks_topic       autoware_perception_msgs/TrackedObjects   the confirmed tracks, for the obstacle adapter
//   tracks_all_topic   autoware_perception_msgs/TrackedObjects   confirmed, tentative and coasting, for the filter
//   markers_topic      visualization_msgs/MarkerArray            box, speed, velocity arrow
// one message each per message received, an empty one included: the filter takes it as "this scan is done". A message
// whose TF does not come, or that is not after the last one, is dropped.
//
// In a TrackedObject: object_id is left empty, the tracks carry no identity; pose and box centre in odom_frame;
// twist.linear is the velocity in odom_frame too (not in the frame of the object as Autoware has it); the x y blocks
// of the pose and twist covariances are those of the Kalman filter, the filter widens its cylinders with them. Every
// parameter is in config/people_tracker.yaml.

#include <memory>
#include <string>

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "people_perception/PeopleTracker.h"

namespace people_perception {

class PeopleTrackerNode : public rclcpp::Node {
 public:
  explicit PeopleTrackerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  using DetectedObjects = autoware_perception_msgs::msg::DetectedObjects;
  using TrackedObjects = autoware_perception_msgs::msg::TrackedObjects;
  using MarkerArray = visualization_msgs::msg::MarkerArray;

  void onDetections(DetectedObjects::ConstSharedPtr msg);
  TrackerParams readTrackerParams();

  std::unique_ptr<PeopleTracker> tracker_;
  std::string odom_frame_;
  double tf_timeout_ = 0.0;
  double marker_lifetime_ = 0.0;  // max_age: a track older than that is gone anyway
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<TrackedObjects>::SharedPtr tracks_publisher_;
  rclcpp::Publisher<TrackedObjects>::SharedPtr tracks_all_publisher_;
  rclcpp::Publisher<MarkerArray>::SharedPtr markers_publisher_;
  rclcpp::Subscription<DetectedObjects>::SharedPtr subscription_;
};

}  // namespace people_perception
