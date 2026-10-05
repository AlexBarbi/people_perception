#include "people_perception/DetectorCloud.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "people_perception/CloudLayout.h"

namespace people_perception {

namespace {

void addField(sensor_msgs::msg::PointCloud2& cloud, const std::string& name, uint32_t offset, uint8_t datatype) {
  sensor_msgs::msg::PointField field;
  field.name = name;
  field.offset = offset;
  field.datatype = datatype;
  field.count = 1;
  cloud.fields.push_back(field);
}

}  // namespace

double yawOf(const Eigen::Quaterniond& odom_from_body) {
  const Eigen::Matrix3d R = odom_from_body.toRotationMatrix();
  return std::atan2(R(1, 0), R(0, 0));
}

Eigen::Matrix3d levelFromBody(const Eigen::Quaterniond& odom_from_body) {
  return Eigen::AngleAxisd(-yawOf(odom_from_body), Eigen::Vector3d::UnitZ()) * odom_from_body.toRotationMatrix();
}

bool toDetectorCloud(const sensor_msgs::msg::PointCloud2& body_cloud, const Eigen::Matrix3d& level_from_body,
                     const DetectorCloudParams& params, const std::string& level_frame,
                     sensor_msgs::msg::PointCloud2& out) {
  using sensor_msgs::msg::PointField;
  const int x_offset = float32Offset(body_cloud, "x");
  const int y_offset = float32Offset(body_cloud, "y");
  const int z_offset = float32Offset(body_cloud, "z");
  if (x_offset < 0 || y_offset < 0 || z_offset < 0 || !hasAllPoints(body_cloud)) {
    return false;
  }
  const int intensity_offset = float32Offset(body_cloud, "intensity");

  out.header.stamp = body_cloud.header.stamp;
  out.header.frame_id = level_frame;
  out.height = 1;
  out.fields.clear();
  addField(out, "x", 0, PointField::FLOAT32);
  addField(out, "y", 4, PointField::FLOAT32);
  addField(out, "z", 8, PointField::FLOAT32);
  addField(out, "intensity", 12, PointField::UINT8);
  addField(out, "return_type", 13, PointField::UINT8);
  addField(out, "channel", 14, PointField::UINT16);
  out.is_bigendian = false;
  out.point_step = kDetectorPointStep;
  out.is_dense = true;
  out.data.clear();
  out.data.reserve(static_cast<size_t>(body_cloud.height) * body_cloud.width * kDetectorPointStep);

  const Eigen::Matrix3f R = level_from_body.cast<float>();
  const float min_range2 = static_cast<float>(params.self_filter_radius * params.self_filter_radius);
  const float max_range2 = static_cast<float>(params.max_range_xy * params.max_range_xy);
  const float z_min = static_cast<float>(params.z_min);
  const float z_max = static_cast<float>(params.z_max);
  for (uint32_t row = 0; row < body_cloud.height; ++row) {
    for (uint32_t col = 0; col < body_cloud.width; ++col) {
      const uint8_t* point = body_cloud.data.data() + static_cast<size_t>(row) * body_cloud.row_step +
                             static_cast<size_t>(col) * body_cloud.point_step;
      Eigen::Vector3f in_body;
      std::memcpy(&in_body.x(), point + x_offset, sizeof(float));
      std::memcpy(&in_body.y(), point + y_offset, sizeof(float));
      std::memcpy(&in_body.z(), point + z_offset, sizeof(float));
      if (!in_body.allFinite()) {
        continue;
      }
      const Eigen::Vector3f p = R * in_body;
      const float range2 = p.x() * p.x() + p.y() * p.y();
      if (range2 < min_range2 || range2 > max_range2 || p.z() < z_min || p.z() > z_max) {
        continue;
      }
      float intensity = 0.0F;
      if (intensity_offset >= 0) {
        std::memcpy(&intensity, point + intensity_offset, sizeof(float));
        intensity *= static_cast<float>(params.intensity_scale);
      }
      // The model reads x y z only (point_feature_size 4): a saturated intensity is enough
      const uint8_t intensity_u8 =
          std::isfinite(intensity) ? static_cast<uint8_t>(std::clamp(std::lround(intensity), 0L, 255L)) : uint8_t{0};

      const size_t end = out.data.size();
      out.data.resize(end + kDetectorPointStep, 0);  // return_type and channel stay 0
      std::memcpy(out.data.data() + end, p.data(), 3 * sizeof(float));
      out.data[end + 12] = intensity_u8;
    }
  }
  out.width = static_cast<uint32_t>(out.data.size() / kDetectorPointStep);
  out.row_step = out.width * kDetectorPointStep;
  return true;
}

}  // namespace people_perception
