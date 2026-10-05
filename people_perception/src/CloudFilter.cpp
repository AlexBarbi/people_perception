#include "people_perception/CloudFilter.h"

#include <cstring>

#include "people_perception/CloudLayout.h"

namespace people_perception {

namespace {

void startLike(const sensor_msgs::msg::PointCloud2& in, sensor_msgs::msg::PointCloud2& out) {
  out.header = in.header;
  out.fields = in.fields;
  out.is_bigendian = in.is_bigendian;
  out.point_step = in.point_step;
  out.is_dense = in.is_dense;
  out.height = 1;
  out.data.clear();
}

void finish(sensor_msgs::msg::PointCloud2& out) {
  out.width = static_cast<uint32_t>(out.data.size() / out.point_step);
  out.row_step = out.width * out.point_step;
}

}  // namespace

std::optional<size_t> removeInsideCylinders(const sensor_msgs::msg::PointCloud2& in,
                                            const std::vector<Cylinder>& cylinders,
                                            const Eigen::Isometry3d& cloud_to_cylinders,
                                            sensor_msgs::msg::PointCloud2& kept,
                                            sensor_msgs::msg::PointCloud2* removed) {
  const int x_offset = float32Offset(in, "x");
  const int y_offset = float32Offset(in, "y");
  const int z_offset = float32Offset(in, "z");
  if (x_offset < 0 || y_offset < 0 || z_offset < 0 || !hasAllPoints(in)) {
    return std::nullopt;
  }
  startLike(in, kept);
  kept.data.reserve(static_cast<size_t>(in.height) * in.width * in.point_step);
  if (removed != nullptr) {
    startLike(in, *removed);
  }

  size_t count = 0;
  for (uint32_t row = 0; row < in.height; ++row) {
    for (uint32_t col = 0; col < in.width; ++col) {
      const uint8_t* point = in.data.data() + static_cast<size_t>(row) * in.row_step + static_cast<size_t>(col) * in.point_step;
      float x, y, z;
      std::memcpy(&x, point + x_offset, sizeof(float));
      std::memcpy(&y, point + y_offset, sizeof(float));
      std::memcpy(&z, point + z_offset, sizeof(float));
      const Eigen::Vector3d p = cloud_to_cylinders * Eigen::Vector3d(x, y, z);
      bool inside = false;
      for (const Cylinder& cylinder : cylinders) {
        const double dx = p.x() - cylinder.x;
        const double dy = p.y() - cylinder.y;
        // A point that is not finite fails every comparison: it stays
        if (dx * dx + dy * dy <= cylinder.radius * cylinder.radius && p.z() >= cylinder.z_min && p.z() <= cylinder.z_max) {
          inside = true;
          break;
        }
      }
      if (!inside) {
        kept.data.insert(kept.data.end(), point, point + in.point_step);
      } else {
        ++count;
        if (removed != nullptr) {
          removed->data.insert(removed->data.end(), point, point + in.point_step);
        }
      }
    }
  }
  finish(kept);
  if (removed != nullptr) {
    finish(*removed);
  }
  return count;
}

}  // namespace people_perception
