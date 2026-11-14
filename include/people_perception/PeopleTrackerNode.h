#pragma once

// ROS 2 node: the PeopleTracker of PeopleTracker.h on the detections of autoware_lidar_centerpoint. For every
// DetectedObjects message it brings the pedestrians to odom_frame with the TF at the stamp of the message, updates the
// tracks and publishes, with that stamp:
//   tracks_topic      autoware_perception_msgs/TrackedObjects   every track, tentative and coasting ones too, in
//                     odom_frame: what dynamic_points_filter takes out of the scans
//   markers_topic     visualization_msgs/MarkerArray            box, speed, velocity arrow, in odom_frame
//   obstacles.topic   legged_controllers/ObstacleArray          with obstacles.enabled: the confirmed tracks within
//                     obstacles.max_range of the sensor, as HUMAN, in obstacles.frame: what OptiPessiController keeps
//                     out of
// one message each per message received, an empty one included: the filter takes it as "this scan is done", and an
// empty obstacle array clears the obstacles of the controller. A message whose TF does not come, or that is not after
// the last one, is dropped.
//
// The controller takes type and position only (radius and maximum speed are those of its type in
// opti_pessi_interface/config/task.info), keeps the closest obstacles for its slots, and takes them in odom only: they
// are brought to obstacles.frame with the latest TF from odom_frame (FAST-LIO publishes odom -> lio_map -> camera_init
// while the legged estimator's /odom runs). That TF only follows the slow drift between the two odometries, and its
// sample for this scan may not be here yet. Without it nothing is published, not an empty array: that would tell the
// controller there is nobody. Each message replaces the obstacles of the previous one: a second publisher of
// obstacles.topic (yolo26_rgbd_detect) erases them.
//
// In a TrackedObject: object_id is left empty, the tracks carry no identity; pose and box centre in odom_frame, no
// heading; twist.linear is the velocity in odom_frame too (not in the frame of the object as Autoware has it); the x y
// blocks of the pose and twist covariances are those of the Kalman filter, the filter widens its cylinders with them.
// Every parameter is in config/people_tracker.yaml.

#include <memory>
#include <string>

#include <Eigen/Core>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <legged_controllers/msg/obstacle_array.hpp>
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
  using ObstacleArray = legged_controllers::msg::ObstacleArray;

  void onDetections(DetectedObjects::ConstSharedPtr msg);
  TrackerParams readTrackerParams();
  /// The confirmed tracks within obstacles.max_range of `sensor` (x y in odom_frame), to the controller.
  void publishObstacles(const builtin_interfaces::msg::Time& stamp, const Eigen::Vector2d& sensor);

  std::unique_ptr<PeopleTracker> tracker_;
  std::string odom_frame_;
  double tf_timeout_ = 0.0;
  double marker_lifetime_ = 0.0;  // max_age: a track older than that is gone anyway
  std::string obstacles_frame_;
  double obstacles_max_range_ = 0.0;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<TrackedObjects>::SharedPtr tracks_publisher_;
  rclcpp::Publisher<MarkerArray>::SharedPtr markers_publisher_;
  rclcpp::Publisher<ObstacleArray>::SharedPtr obstacles_publisher_;  // obstacles.enabled only
  rclcpp::Subscription<DetectedObjects>::SharedPtr subscription_;
};

}  // namespace people_perception
