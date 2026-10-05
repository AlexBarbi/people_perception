#include "people_perception/DetectorCloudAdapter.h"

#include <cmath>
#include <utility>

#include <geometry_msgs/msg/transform_stamped.hpp>

namespace people_perception {

namespace {

int64_t stampOf(const std_msgs::msg::Header& header) { return rclcpp::Time(header.stamp).nanoseconds(); }

}  // namespace

DetectorCloudAdapter::DetectorCloudAdapter(const rclcpp::NodeOptions& options)
    : Node("detector_cloud_adapter", options) {
  level_frame_ = declare_parameter<std::string>("level_frame");
  odometry_history_ = static_cast<size_t>(declare_parameter<int>("odometry_history"));
  cloud_queue_ = static_cast<size_t>(declare_parameter<int>("cloud_queue"));
  params_.self_filter_radius = declare_parameter<double>("self_filter_radius");
  params_.max_range_xy = declare_parameter<double>("max_range_xy");
  params_.z_min = declare_parameter<double>("z_min");
  params_.z_max = declare_parameter<double>("z_max");
  params_.intensity_scale = declare_parameter<double>("intensity_scale");

  // Reliable on both sides, as FAST-LIO publishes: its clouds are about 1 MB and best effort loses them to fragment
  // drops
  const rclcpp::QoS qos(static_cast<size_t>(declare_parameter<int>("qos_depth")));
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  publisher_ = create_publisher<PointCloud2>(declare_parameter<std::string>("output_cloud_topic"), qos);
  odometry_subscription_ = create_subscription<Odometry>(
      declare_parameter<std::string>("odometry_topic"), qos,
      [this](Odometry::ConstSharedPtr odometry) { onOdometry(std::move(odometry)); });
  cloud_subscription_ = create_subscription<PointCloud2>(
      declare_parameter<std::string>("input_cloud_topic"), qos,
      [this](PointCloud2::ConstSharedPtr cloud) { onCloud(std::move(cloud)); });
}

void DetectorCloudAdapter::onCloud(PointCloud2::ConstSharedPtr cloud) {
  waiting_.push_back(std::move(cloud));
  if (waiting_.size() > cloud_queue_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "%zu scans wait for their odometry: is it still coming? The oldest is dropped", cloud_queue_);
    waiting_.pop_front();
  }
  release();
}

void DetectorCloudAdapter::onOdometry(Odometry::ConstSharedPtr odometry) {
  const int64_t stamp = stampOf(odometry->header);
  odometry_[stamp] = std::move(odometry);
  while (odometry_.size() > odometry_history_) {
    odometry_.erase(odometry_.begin());
  }
  release();
}

void DetectorCloudAdapter::release() {
  while (!waiting_.empty()) {
    const int64_t stamp = stampOf(waiting_.front()->header);
    const auto odometry = odometry_.find(stamp);
    if (odometry != odometry_.end()) {
      convert(*waiting_.front(), *odometry->second);
    } else if (!odometry_.empty() && odometry_.rbegin()->first > stamp) {
      // The odometry messages come in order: this scan has none
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "No odometry with the stamp %.3f of a scan: scan dropped",
                           static_cast<double>(stamp) * 1e-9);
    } else {
      return;  // its odometry is still on the way
    }
    waiting_.pop_front();
  }
}

void DetectorCloudAdapter::convert(const PointCloud2& cloud, const Odometry& odometry) {
  if (cloud.header.frame_id != odometry.child_frame_id) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "The cloud is in %s but the odometry is the pose of %s",
                         cloud.header.frame_id.c_str(), odometry.child_frame_id.c_str());
  }
  const auto& orientation = odometry.pose.pose.orientation;
  const Eigen::Quaterniond odom_from_body(orientation.w, orientation.x, orientation.y, orientation.z);
  auto out = std::make_unique<PointCloud2>();
  if (!toDetectorCloud(cloud, levelFromBody(odom_from_body), params_, level_frame_, *out)) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                          "Cloud without FLOAT32 x y z fields or with short data: scan dropped");
    return;
  }

  // The transform first: who gets the cloud looks it up at the stamp of the cloud
  const double yaw = yawOf(odom_from_body);
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = cloud.header.stamp;
  transform.header.frame_id = odometry.header.frame_id;
  transform.child_frame_id = level_frame_;
  transform.transform.translation.x = odometry.pose.pose.position.x;
  transform.transform.translation.y = odometry.pose.pose.position.y;
  transform.transform.translation.z = odometry.pose.pose.position.z;
  transform.transform.rotation.z = std::sin(yaw / 2.0);
  transform.transform.rotation.w = std::cos(yaw / 2.0);
  tf_broadcaster_->sendTransform(transform);
  publisher_->publish(std::move(out));
}

}  // namespace people_perception
