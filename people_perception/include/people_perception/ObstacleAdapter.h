#pragma once

// ROS 2 node: block B10 of docs/perception_pipeline.md. The confirmed tracks of people_tracker as what an
// Optimistic-Pessimistic MPC needs of each person: where they are, a radius, and the speed their class can reach.
//   obstacles_topic   people_perception_msgs/DynamicObstacleArray   one per tracks message, an empty one included:
//                     a consumer that stops receiving it knows the perception has stopped
//   markers_topic     visualization_msgs/MarkerArray                two circles per obstacle: its radius, and
//                     radius + max_speed * marker_horizon, the room the pessimistic constraint takes at the end of
//                     the horizon
// Stamp and frame are those of the tracks: the instant of the measurement (end of the scan), in the FAST-LIO odometry
// frame. Positions are not moved on and radii not widened for the latency: the consumer adds
// max_speed * (time of use - stamp) to each radius.
//
// Of the people within max_range of the robot (its odometry at the stamp of the tracks, else the last one), the
// max_obstacles closest, closest first. max_speed comes from the class, never from the estimated velocity: a person
// standing still can start walking.
//
// It does not feed the Opti-Pessi controller of this workspace, which reads /opti_pessi/obstacles
// (legged_controllers/ObstacleArray, frame odom): see RICOGNIZIONE.md. Every parameter is in
// config/obstacle_adapter.yaml.

#include <cstdint>
#include <map>
#include <string>

#include <Eigen/Core>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <people_perception_msgs/msg/dynamic_obstacle_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace people_perception {

// From config/obstacle_adapter.yaml, see there.
struct ObstacleParams {
  double max_range = 0.0;
  int max_obstacles = 0;
  double radius_min = 0.0;
  double radius_max = 0.0;
  double radius_margin = 0.0;
  double max_speed = 0.0;
};

/// The pedestrians of `tracks` within max_range of `robot` (x y in the frame of the tracks) as obstacles, closest
/// first, max_obstacles at most, with the header of the tracks.
people_perception_msgs::msg::DynamicObstacleArray toObstacles(
    const autoware_perception_msgs::msg::TrackedObjects& tracks, const Eigen::Vector2d& robot,
    const ObstacleParams& params);

class ObstacleAdapter : public rclcpp::Node {
 public:
  explicit ObstacleAdapter(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  using TrackedObjects = autoware_perception_msgs::msg::TrackedObjects;
  using Odometry = nav_msgs::msg::Odometry;
  using DynamicObstacleArray = people_perception_msgs::msg::DynamicObstacleArray;
  using MarkerArray = visualization_msgs::msg::MarkerArray;

  void onTracks(const TrackedObjects& tracks);
  void onOdometry(const Odometry& odometry);
  MarkerArray markersOf(const DynamicObstacleArray& obstacles) const;

  ObstacleParams params_;
  double marker_horizon_ = 0.0;
  double marker_lifetime_ = 0.0;
  size_t odometry_history_ = 0;
  std::string odometry_frame_;
  std::map<int64_t, Eigen::Vector2d> robot_;  // position of the robot by stamp [ns]
  rclcpp::Publisher<DynamicObstacleArray>::SharedPtr obstacles_publisher_;
  rclcpp::Publisher<MarkerArray>::SharedPtr markers_publisher_;
  rclcpp::Subscription<TrackedObjects>::SharedPtr tracks_subscription_;
  rclcpp::Subscription<Odometry>::SharedPtr odometry_subscription_;
};

}  // namespace people_perception
