#pragma once

// ROS 2 node: block B3 of docs/perception_pipeline.md. For every scan of FAST-LIO it publishes
//   output_cloud_topic   the body cloud as autoware_lidar_centerpoint takes it (see DetectorCloud.h), in level_frame
//   TF                   odometry frame -> level_frame: the position of the body and its yaw alone
// both with the stamp of the scan. The tracker brings the detections back to the odometry frame with that TF.
//
// A cloud is paired with the odometry message of the same stamp (FAST-LIO stamps the two alike). The two travel on
// their own and the small odometry message can come after the next cloud: clouds wait for theirs in order, cloud_queue
// of them at most. A scan is dropped when a later odometry is here and its own is not: it will not come any more.
// Every parameter is in config/detector_cloud_adapter.yaml, none has a default here.

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "people_perception/DetectorCloud.h"

namespace people_perception {

class DetectorCloudAdapter : public rclcpp::Node {
 public:
  explicit DetectorCloudAdapter(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  using PointCloud2 = sensor_msgs::msg::PointCloud2;
  using Odometry = nav_msgs::msg::Odometry;

  void onCloud(PointCloud2::ConstSharedPtr cloud);
  void onOdometry(Odometry::ConstSharedPtr odometry);
  /// Convert, in order, every waiting cloud whose odometry is here; drop those whose odometry will not come.
  void release();
  void convert(const PointCloud2& cloud, const Odometry& odometry);

  DetectorCloudParams params_;
  std::string level_frame_;
  size_t odometry_history_ = 0;
  size_t cloud_queue_ = 0;
  std::map<int64_t, Odometry::ConstSharedPtr> odometry_;  // by stamp [ns]
  std::deque<PointCloud2::ConstSharedPtr> waiting_;       // the clouds whose odometry has not come, oldest first
  rclcpp::Publisher<PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<PointCloud2>::SharedPtr cloud_subscription_;
  rclcpp::Subscription<Odometry>::SharedPtr odometry_subscription_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace people_perception
