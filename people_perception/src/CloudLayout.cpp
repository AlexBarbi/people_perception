#include "people_perception/CloudLayout.h"

namespace people_perception {

int float32Offset(const sensor_msgs::msg::PointCloud2& cloud, const std::string& name) {
  for (const auto& field : cloud.fields) {
    if (field.name == name && field.datatype == sensor_msgs::msg::PointField::FLOAT32 && field.count == 1 &&
        field.offset + sizeof(float) <= cloud.point_step) {
      return static_cast<int>(field.offset);
    }
  }
  return -1;
}

bool hasAllPoints(const sensor_msgs::msg::PointCloud2& cloud) {
  if (cloud.height == 0 || cloud.width == 0) {
    return true;
  }
  return cloud.row_step >= static_cast<size_t>(cloud.width) * cloud.point_step &&
         cloud.data.size() >=
             static_cast<size_t>(cloud.height - 1) * cloud.row_step + static_cast<size_t>(cloud.width) * cloud.point_step;
}

}  // namespace people_perception
