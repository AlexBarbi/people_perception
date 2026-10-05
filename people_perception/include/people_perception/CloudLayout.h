#pragma once

// Reading a PointCloud2 by its bytes, whatever its point type.

#include <string>

#include <sensor_msgs/msg/point_cloud2.hpp>

namespace people_perception {

/// Offset of the FLOAT32 field `name` (one value per point) in a point, or -1 if the cloud has none.
int float32Offset(const sensor_msgs::msg::PointCloud2& cloud, const std::string& name);

/// True if `data` holds the height * width points the cloud declares.
bool hasAllPoints(const sensor_msgs::msg::PointCloud2& cloud);

}  // namespace people_perception
