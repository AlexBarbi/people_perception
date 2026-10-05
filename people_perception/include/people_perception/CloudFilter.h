#pragma once

// Removal of the points of the tracked people from a cloud (block B6 of docs/perception_pipeline.md), on the bytes of
// the PointCloud2: the points that stay keep every byte, whatever the layout, so the result has the fields, the
// point_step, the frame and the stamp of the input.

#include <cstddef>
#include <optional>
#include <vector>

#include <Eigen/Geometry>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace people_perception {

/// Vertical cylinder around a person, in the frame the tracks are in.
struct Cylinder {
  double x = 0.0;
  double y = 0.0;
  double radius = 0.0;
  double z_min = 0.0;
  double z_max = 0.0;
};

/// `in` without the points inside a cylinder, in `kept`; the points taken out in `removed`, when given. Both get the
/// header and the layout of `in`, as unorganized clouds. A point p of the cloud is tested at cloud_to_cylinders * p
/// (identity for a cloud in the frame of the cylinders). Points that are not finite are kept.
/// Returns the number of points removed, or nothing if the cloud has no FLOAT32 x y z fields or too little data.
std::optional<size_t> removeInsideCylinders(const sensor_msgs::msg::PointCloud2& in,
                                            const std::vector<Cylinder>& cylinders,
                                            const Eigen::Isometry3d& cloud_to_cylinders,
                                            sensor_msgs::msg::PointCloud2& kept,
                                            sensor_msgs::msg::PointCloud2* removed = nullptr);

}  // namespace people_perception
