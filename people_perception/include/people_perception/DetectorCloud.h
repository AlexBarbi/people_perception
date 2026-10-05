#pragma once

// The cloud autoware_lidar_centerpoint takes, from the body cloud of FAST-LIO (block B3 of
// docs/perception_pipeline.md):
//  - layout PointXYZIRC of autoware_point_types, 16 bytes: x y z FLOAT32, intensity UINT8 at 12, return_type UINT8 at
//    13, channel UINT16 at 14. The node drops any other layout ("Expected PointXYZIRC").
//  - frame "sensor level": the origin of the body frame, its yaw, no roll and no pitch. A pillar detector assumes a
//    level floor at a constant height, and the body of a trotting quadruped rolls and pitches.

#include <string>

#include <Eigen/Geometry>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace people_perception {

inline constexpr uint32_t kDetectorPointStep = 16;

// From config/detector_cloud_adapter.yaml, see there.
struct DetectorCloudParams {
  double self_filter_radius = 0.0;
  double max_range_xy = 0.0;
  double z_min = 0.0;
  double z_max = 0.0;
  double intensity_scale = 0.0;
};

/// Yaw of a body orientation: the heading of its x axis on the horizontal plane of the odometry frame.
double yawOf(const Eigen::Quaterniond& odom_from_body);

/// Rotation from the body frame to the sensor level frame: Rz(-yaw) * R_odom_body.
Eigen::Matrix3d levelFromBody(const Eigen::Quaterniond& odom_from_body);

/// `body_cloud` (FLOAT32 x y z, and intensity if it has one) as a detector cloud in `level_frame`, same stamp, without
/// the points that are not finite, closer than self_filter_radius to the vertical through the sensor, farther than
/// max_range_xy from it or outside z_min .. z_max. return_type and channel are 0.
/// False if the cloud has no FLOAT32 x y z fields or too little data.
bool toDetectorCloud(const sensor_msgs::msg::PointCloud2& body_cloud, const Eigen::Matrix3d& level_from_body,
                     const DetectorCloudParams& params, const std::string& level_frame,
                     sensor_msgs::msg::PointCloud2& out);

}  // namespace people_perception
