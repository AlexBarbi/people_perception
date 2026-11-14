#include "people_perception/PeopleTrackerNode.h"

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

#include <tf2/exceptions.h>
#include <tf2_eigen/tf2_eigen.hpp>

namespace people_perception {

namespace {

using autoware_perception_msgs::msg::ObjectClassification;
using autoware_perception_msgs::msg::Shape;
using autoware_perception_msgs::msg::TrackedObject;
using legged_controllers::msg::Obstacle;
using visualization_msgs::msg::Marker;

bool isPedestrian(const autoware_perception_msgs::msg::DetectedObject& object) {
  const auto best = std::max_element(
      object.classification.begin(), object.classification.end(),
      [](const ObjectClassification& a, const ObjectClassification& b) { return a.probability < b.probability; });
  return best != object.classification.end() && best->label == ObjectClassification::PEDESTRIAN;
}

TrackedObject toMessage(const Track& track) {
  TrackedObject object;
  object.existence_probability = static_cast<float>(track.score);
  ObjectClassification classification;
  classification.label = ObjectClassification::PEDESTRIAN;
  classification.probability = 1.0F;
  object.classification.push_back(classification);

  auto& pose = object.kinematics.pose_with_covariance;
  pose.pose.position.x = track.position().x();
  pose.pose.position.y = track.position().y();
  pose.pose.position.z = track.z;
  pose.pose.orientation.w = 1.0;  // no heading: the box of a pedestrian says little, and not which way they face
  pose.covariance[0] = track.covariance(0, 0);
  pose.covariance[1] = pose.covariance[6] = track.covariance(0, 1);
  pose.covariance[7] = track.covariance(1, 1);
  auto& twist = object.kinematics.twist_with_covariance;
  twist.twist.linear.x = track.velocity().x();
  twist.twist.linear.y = track.velocity().y();
  twist.covariance[0] = track.covariance(2, 2);
  twist.covariance[1] = twist.covariance[6] = track.covariance(2, 3);
  twist.covariance[7] = track.covariance(3, 3);

  object.shape.type = Shape::BOUNDING_BOX;
  object.shape.dimensions.x = track.dimensions.x();
  object.shape.dimensions.y = track.dimensions.y();
  object.shape.dimensions.z = track.dimensions.z();
  return object;
}

}  // namespace

PeopleTrackerNode::PeopleTrackerNode(const rclcpp::NodeOptions& options) : Node("people_tracker", options) {
  const TrackerParams params = readTrackerParams();
  tracker_ = std::make_unique<PeopleTracker>(params);
  marker_lifetime_ = params.max_age;
  odom_frame_ = declare_parameter<std::string>("odom_frame");
  tf_timeout_ = declare_parameter<double>("tf_timeout");
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  const rclcpp::QoS qos(static_cast<size_t>(declare_parameter<int>("qos_depth")));
  tracks_publisher_ = create_publisher<TrackedObjects>(declare_parameter<std::string>("tracks_topic"), qos);
  markers_publisher_ = create_publisher<MarkerArray>(declare_parameter<std::string>("markers_topic"), qos);
  // Read with enabled false too: a value missing from the yaml shows now, not the day the obstacles are switched on
  const bool obstacles = declare_parameter<bool>("obstacles.enabled");
  const std::string obstacles_topic = declare_parameter<std::string>("obstacles.topic");
  obstacles_frame_ = declare_parameter<std::string>("obstacles.frame");
  obstacles_max_range_ = declare_parameter<double>("obstacles.max_range");
  if (obstacles) {
    obstacles_publisher_ = create_publisher<ObstacleArray>(obstacles_topic, rclcpp::QoS(1));  // as the controller
  }
  subscription_ = create_subscription<DetectedObjects>(
      declare_parameter<std::string>("detections_topic"), qos,
      [this](DetectedObjects::ConstSharedPtr msg) { onDetections(std::move(msg)); });
}

TrackerParams PeopleTrackerNode::readTrackerParams() {
  TrackerParams params;
  params.gate_distance = declare_parameter<double>("gate_distance");
  params.max_plausible_speed = declare_parameter<double>("max_plausible_speed");
  params.min_score_new_track = declare_parameter<double>("min_score_new_track");
  params.min_score_confirm = declare_parameter<double>("min_score_confirm");
  params.min_hits = static_cast<int>(declare_parameter<int>("min_hits"));
  params.hits_window = static_cast<int>(declare_parameter<int>("hits_window"));
  params.max_age = declare_parameter<double>("max_age");
  params.shape_smoothing = declare_parameter<double>("shape_smoothing");
  params.process_noise_acceleration = declare_parameter<double>("process_noise_acceleration");
  params.measurement_noise_position = declare_parameter<double>("measurement_noise_position");
  params.initial_speed_std = declare_parameter<double>("initial_speed_std");
  params.backward_jump_reset = declare_parameter<double>("backward_jump_reset");
  return params;
}

void PeopleTrackerNode::onDetections(DetectedObjects::ConstSharedPtr msg) {
  Eigen::Isometry3d odom_from_detector;
  try {
    odom_from_detector = tf2::transformToEigen(tf_buffer_->lookupTransform(
        odom_frame_, msg->header.frame_id, rclcpp::Time(msg->header.stamp), rclcpp::Duration::from_seconds(tf_timeout_)));
  } catch (const tf2::TransformException& e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "No TF %s <- %s at the stamp of the detections, dropped: %s",
                         odom_frame_.c_str(), msg->header.frame_id.c_str(), e.what());
    return;
  }

  std::vector<Detection> detections;
  for (const auto& object : msg->objects) {
    if (!isPedestrian(object)) {
      continue;
    }
    const auto& pose = object.kinematics.pose_with_covariance.pose;
    Detection detection;
    detection.position = odom_from_detector * Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
    detection.dimensions = Eigen::Vector3d(object.shape.dimensions.x, object.shape.dimensions.y, object.shape.dimensions.z);
    detection.score = object.existence_probability;
    detections.push_back(detection);
  }

  const rclcpp::Time stamp(msg->header.stamp);
  if (!tracker_->update(stamp.seconds(), detections)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Detections at %.3f s are not after the last ones taken, dropped", stamp.seconds());
    return;
  }

  TrackedObjects tracks;
  tracks.header.stamp = msg->header.stamp;
  tracks.header.frame_id = odom_frame_;
  MarkerArray markers;
  Marker clear;  // the tracks of the last message: there may be fewer now
  clear.header = tracks.header;
  clear.action = Marker::DELETEALL;
  markers.markers.push_back(clear);
  int id = 0;
  for (const Track& track : tracker_->tracks()) {
    tracks.objects.push_back(toMessage(track));

    // Confirmed green, tentative yellow, fainter while coasting on the prediction
    Marker box;
    box.header = tracks.header;
    box.ns = "box";
    box.id = id;
    box.type = Marker::CUBE;
    box.pose = tracks.objects.back().kinematics.pose_with_covariance.pose;
    box.scale = tracks.objects.back().shape.dimensions;
    box.color.r = track.confirmed ? 0.1F : 0.9F;
    box.color.g = 0.8F;
    box.color.b = 0.1F;
    box.color.a = track.time_since_update > 0.0 ? 0.25F : 0.6F;
    box.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_);  // gone by itself if the tracker stops
    markers.markers.push_back(box);

    Marker label;
    label.header = tracks.header;
    label.ns = "speed";
    label.id = id;
    label.type = Marker::TEXT_VIEW_FACING;
    label.pose.position = box.pose.position;
    label.pose.position.z += track.dimensions.z() / 2.0 + 0.2;
    label.pose.orientation.w = 1.0;
    label.scale.z = 0.25;
    label.color.r = label.color.g = label.color.b = label.color.a = 1.0F;
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f m/s", track.velocity().norm());
    label.text = text;
    label.lifetime = box.lifetime;
    markers.markers.push_back(label);

    Marker arrow;  // where the person is in one second
    arrow.header = tracks.header;
    arrow.ns = "velocity";
    arrow.id = id;
    arrow.type = Marker::ARROW;
    arrow.pose.orientation.w = 1.0;
    arrow.points.resize(2);
    arrow.points[0] = box.pose.position;
    arrow.points[1] = box.pose.position;
    arrow.points[1].x += track.velocity().x();
    arrow.points[1].y += track.velocity().y();
    arrow.scale.x = 0.05;
    arrow.scale.y = 0.12;
    arrow.color = box.color;
    arrow.color.a = 0.9F;
    arrow.lifetime = box.lifetime;
    markers.markers.push_back(arrow);
    ++id;
  }
  tracks_publisher_->publish(tracks);
  markers_publisher_->publish(markers);
  if (obstacles_publisher_) {
    // The detections are in the frame of the sensor: its origin is where the sensor is
    publishObstacles(msg->header.stamp, odom_from_detector.translation().head<2>());
  }
}

void PeopleTrackerNode::publishObstacles(const builtin_interfaces::msg::Time& stamp, const Eigen::Vector2d& sensor) {
  Eigen::Isometry3d obstacles_from_odom;
  try {
    obstacles_from_odom =
        tf2::transformToEigen(tf_buffer_->lookupTransform(obstacles_frame_, odom_frame_, tf2::TimePointZero));
  } catch (const tf2::TransformException& e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "No TF %s <- %s, no obstacles for the controller: %s",
                         obstacles_frame_.c_str(), odom_frame_.c_str(), e.what());
    return;
  }
  ObstacleArray obstacles;
  obstacles.header.stamp = stamp;
  obstacles.header.frame_id = obstacles_frame_;
  for (const Track& track : tracker_->tracks()) {
    // Each obstacle takes a slot of the controller, and its solve time grows with them
    if (!track.confirmed || (track.position() - sensor).norm() > obstacles_max_range_) {
      continue;
    }
    const Eigen::Vector3d position = obstacles_from_odom * Eigen::Vector3d(track.state.x(), track.state.y(), track.z);
    Obstacle obstacle;
    obstacle.type = Obstacle::HUMAN;
    obstacle.position.x = position.x();
    obstacle.position.y = position.y();
    obstacle.position.z = position.z();
    obstacles.obstacles.push_back(obstacle);
  }
  if (count_publishers(obstacles_publisher_->get_topic_name()) > 1) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                         "%s has another publisher: each message replaces the obstacles of the other one",
                         obstacles_publisher_->get_topic_name());
  }
  obstacles_publisher_->publish(obstacles);
}

}  // namespace people_perception
