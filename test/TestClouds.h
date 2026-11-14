#pragma once

#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace test {

struct Xyzi {
  float x, y, z, intensity;
};

inline void addField(sensor_msgs::msg::PointCloud2& cloud, const std::string& name, uint32_t offset,
                     uint8_t datatype = sensor_msgs::msg::PointField::FLOAT32) {
  sensor_msgs::msg::PointField field;
  field.name = name;
  field.offset = offset;
  field.datatype = datatype;
  field.count = 1;
  cloud.fields.push_back(field);
}

// A cloud as FAST-LIO publishes it (pcl::PointXYZINormal, 48 bytes per point with padding after z, normal_z and
// curvature). The bytes that are not x y z intensity are random, the padding too: only a copy of the bytes gets a
// point through unchanged.
inline sensor_msgs::msg::PointCloud2 makeFastLioCloud(const std::vector<Xyzi>& points,
                                                      const std::string& frame = "rslidar_lio", int32_t sec = 10,
                                                      uint32_t nanosec = 699999999, uint32_t height = 1) {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = frame;
  cloud.header.stamp.sec = sec;
  cloud.header.stamp.nanosec = nanosec;
  cloud.height = height;
  cloud.width = static_cast<uint32_t>(points.size()) / height;
  addField(cloud, "x", 0);
  addField(cloud, "y", 4);
  addField(cloud, "z", 8);
  addField(cloud, "normal_x", 16);
  addField(cloud, "normal_y", 20);
  addField(cloud, "normal_z", 24);
  addField(cloud, "intensity", 32);
  addField(cloud, "curvature", 36);
  cloud.is_bigendian = false;
  cloud.point_step = 48;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.is_dense = true;
  cloud.data.resize(points.size() * cloud.point_step);
  std::mt19937 random(11 + static_cast<uint32_t>(sec) + nanosec);
  for (auto& byte : cloud.data) {
    byte = static_cast<uint8_t>(random());
  }
  for (size_t i = 0; i < points.size(); ++i) {
    uint8_t* point = cloud.data.data() + i * cloud.point_step;
    std::memcpy(point + 0, &points[i].x, 4);
    std::memcpy(point + 4, &points[i].y, 4);
    std::memcpy(point + 8, &points[i].z, 4);
    std::memcpy(point + 32, &points[i].intensity, 4);
  }
  return cloud;
}

// The bytes of the given points of a cloud, one after the other.
inline std::vector<uint8_t> bytesOf(const sensor_msgs::msg::PointCloud2& cloud, const std::vector<size_t>& points) {
  std::vector<uint8_t> bytes;
  for (size_t i : points) {
    const auto begin = cloud.data.begin() + static_cast<std::ptrdiff_t>(i * cloud.point_step);
    bytes.insert(bytes.end(), begin, begin + cloud.point_step);
  }
  return bytes;
}

// The odometry FAST-LIO publishes with a scan: the pose of rslidar_lio in camera_init, same stamp as the clouds.
inline nav_msgs::msg::Odometry makeOdometry(int32_t sec, uint32_t nanosec,
                                            const Eigen::Vector3d& position = Eigen::Vector3d::Zero(),
                                            const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity()) {
  nav_msgs::msg::Odometry odometry;
  odometry.header.frame_id = "camera_init";
  odometry.header.stamp.sec = sec;
  odometry.header.stamp.nanosec = nanosec;
  odometry.child_frame_id = "rslidar_lio";
  odometry.pose.pose.position.x = position.x();
  odometry.pose.pose.position.y = position.y();
  odometry.pose.pose.position.z = position.z();
  odometry.pose.pose.orientation.x = orientation.x();
  odometry.pose.pose.orientation.y = orientation.y();
  odometry.pose.pose.orientation.z = orientation.z();
  odometry.pose.pose.orientation.w = orientation.w();
  return odometry;
}

}  // namespace test
