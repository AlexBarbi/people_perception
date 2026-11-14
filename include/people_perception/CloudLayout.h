#pragma once

// The messages of a scan: their stamp, and a PointCloud2 read by its bytes, whatever its point type.

#include <cstdint>
#include <string>

#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

namespace people_perception {

/// Stamp of a message [ns]: the key that pairs the clouds, the odometry and the tracks of a scan.
inline int64_t stampOf(const std_msgs::msg::Header& header) { return rclcpp::Time(header.stamp).nanoseconds(); }

/// Offset of the FLOAT32 field `name` (one value per point) in a point, or -1 if the cloud has none.
int float32Offset(const sensor_msgs::msg::PointCloud2& cloud, const std::string& name);

/// True if `data` holds the height * width points the cloud declares.
bool hasAllPoints(const sensor_msgs::msg::PointCloud2& cloud);

/// An unorganized cloud without points, with the header and the layout of `like`.
sensor_msgs::msg::PointCloud2 emptyLike(const sensor_msgs::msg::PointCloud2& like);

}  // namespace people_perception
