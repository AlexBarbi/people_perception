#include "people_perception/ObstacleAdapter.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace people_perception {

namespace {

using autoware_perception_msgs::msg::ObjectClassification;
using people_perception_msgs::msg::DynamicObstacle;
using visualization_msgs::msg::Marker;

bool isPedestrian(const autoware_perception_msgs::msg::TrackedObject& track) {
  const auto best = std::max_element(
      track.classification.begin(), track.classification.end(),
      [](const ObjectClassification& a, const ObjectClassification& b) { return a.probability < b.probability; });
  return best != track.classification.end() && best->label == ObjectClassification::PEDESTRIAN;
}

// A horizontal circle around an obstacle.
Marker circle(const std_msgs::msg::Header& header, const DynamicObstacle& obstacle, const std::string& ns, int id,
              double radius, float alpha) {
  Marker marker;
  marker.header = header;
  marker.ns = ns;
  marker.id = id;
  marker.type = Marker::LINE_STRIP;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.03;
  marker.color.r = 1.0F;
  marker.color.g = 0.45F;
  marker.color.b = 0.1F;
  marker.color.a = alpha;
  const int segments = 48;
  for (int i = 0; i <= segments; ++i) {
    const double angle = 2.0 * M_PI * i / segments;
    geometry_msgs::msg::Point point;
    point.x = obstacle.position.x + radius * std::cos(angle);
    point.y = obstacle.position.y + radius * std::sin(angle);
    point.z = obstacle.position.z;
    marker.points.push_back(point);
  }
  return marker;
}

}  // namespace

people_perception_msgs::msg::DynamicObstacleArray toObstacles(
    const autoware_perception_msgs::msg::TrackedObjects& tracks, const Eigen::Vector2d& robot,
    const ObstacleParams& params) {
  std::vector<std::pair<double, DynamicObstacle>> in_range;
  for (const auto& track : tracks.objects) {
    if (!isPedestrian(track)) {
      continue;  // the only class with a radius and a speed so far
    }
    const auto& position = track.kinematics.pose_with_covariance.pose.position;
    const double distance = std::hypot(position.x - robot.x(), position.y - robot.y());
    if (distance > params.max_range) {
      continue;
    }
    DynamicObstacle obstacle;
    obstacle.id = track.object_id;
    obstacle.label = ObjectClassification::PEDESTRIAN;
    obstacle.position = position;
    const double half_side = std::max(track.shape.dimensions.x, track.shape.dimensions.y) / 2.0;
    obstacle.radius =
        static_cast<float>(std::clamp(half_side + params.radius_margin, params.radius_min, params.radius_max));
    // Of the class, not of the track: the pessimistic branch covers whatever the person may do, standing ones too
    obstacle.max_speed = static_cast<float>(params.max_speed);
    obstacle.velocity = track.kinematics.twist_with_covariance.twist.linear;
    obstacle.is_stationary = track.kinematics.is_stationary;
    obstacle.confidence = track.existence_probability;
    in_range.emplace_back(distance, obstacle);
  }
  std::stable_sort(in_range.begin(), in_range.end(),
                   [](const auto& a, const auto& b) { return a.first < b.first; });

  people_perception_msgs::msg::DynamicObstacleArray out;
  out.header = tracks.header;
  for (size_t i = 0; i < in_range.size() && i < static_cast<size_t>(params.max_obstacles); ++i) {
    out.obstacles.push_back(in_range[i].second);
  }
  return out;
}

ObstacleAdapter::ObstacleAdapter(const rclcpp::NodeOptions& options) : Node("obstacle_adapter", options) {
  params_.max_range = declare_parameter<double>("max_range");
  params_.max_obstacles = static_cast<int>(declare_parameter<int>("max_obstacles"));
  params_.radius_margin = declare_parameter<double>("pedestrian.radius_margin");
  params_.radius_min = declare_parameter<double>("pedestrian.radius_min");
  params_.radius_max = declare_parameter<double>("pedestrian.radius_max");
  params_.max_speed = declare_parameter<double>("pedestrian.max_speed");
  marker_horizon_ = declare_parameter<double>("marker_horizon");
  marker_lifetime_ = declare_parameter<double>("marker_lifetime");
  odometry_history_ = static_cast<size_t>(declare_parameter<int>("odometry_history"));

  const rclcpp::QoS qos(static_cast<size_t>(declare_parameter<int>("qos_depth")));
  obstacles_publisher_ = create_publisher<DynamicObstacleArray>(declare_parameter<std::string>("obstacles_topic"), qos);
  markers_publisher_ = create_publisher<MarkerArray>(declare_parameter<std::string>("markers_topic"), qos);
  odometry_subscription_ = create_subscription<Odometry>(
      declare_parameter<std::string>("odometry_topic"), qos,
      [this](Odometry::ConstSharedPtr odometry) { onOdometry(*odometry); });
  tracks_subscription_ = create_subscription<TrackedObjects>(
      declare_parameter<std::string>("tracks_topic"), qos,
      [this](TrackedObjects::ConstSharedPtr tracks) { onTracks(*tracks); });
}

void ObstacleAdapter::onOdometry(const Odometry& odometry) {
  odometry_frame_ = odometry.header.frame_id;
  robot_[rclcpp::Time(odometry.header.stamp).nanoseconds()] =
      Eigen::Vector2d(odometry.pose.pose.position.x, odometry.pose.pose.position.y);
  while (robot_.size() > odometry_history_) {
    robot_.erase(robot_.begin());
  }
}

void ObstacleAdapter::onTracks(const TrackedObjects& tracks) {
  if (robot_.empty()) {
    // Which people are in range and which are the closest cannot be told: a silent topic says so to the consumer
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "No odometry yet: no obstacles published");
    return;
  }
  if (odometry_frame_ != tracks.header.frame_id) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                          "The tracks are in %s, the odometry in %s: no obstacles published",
                          tracks.header.frame_id.c_str(), odometry_frame_.c_str());
    return;
  }
  // The robot at the scan of these tracks; its last known position if that odometry is not here
  const auto at_stamp = robot_.find(rclcpp::Time(tracks.header.stamp).nanoseconds());
  const Eigen::Vector2d robot = at_stamp != robot_.end() ? at_stamp->second : robot_.rbegin()->second;

  const DynamicObstacleArray obstacles = toObstacles(tracks, robot, params_);
  obstacles_publisher_->publish(obstacles);
  markers_publisher_->publish(markersOf(obstacles));
}

ObstacleAdapter::MarkerArray ObstacleAdapter::markersOf(const DynamicObstacleArray& obstacles) const {
  MarkerArray markers;
  Marker clear;  // the obstacles of the last message: there may be fewer now
  clear.header = obstacles.header;
  clear.action = Marker::DELETEALL;
  markers.markers.push_back(clear);
  int id = 0;
  for (const auto& obstacle : obstacles.obstacles) {
    // The obstacle itself, and how far the pessimistic constraint keeps the robot at the end of the horizon
    for (Marker marker : {circle(obstacles.header, obstacle, "radius", id, obstacle.radius, 1.0F),
                          circle(obstacles.header, obstacle, "pessimistic", id,
                                 obstacle.radius + obstacle.max_speed * marker_horizon_, 0.5F)}) {
      marker.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);
      markers.markers.push_back(marker);
    }
    ++id;
  }
  return markers;
}

}  // namespace people_perception
