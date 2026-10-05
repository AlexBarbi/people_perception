#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "people_perception/CloudFilter.h"
#include "people_perception/DetectorCloud.h"

#include "TestClouds.h"

using people_perception::Cylinder;
using people_perception::DetectorCloudParams;
using people_perception::levelFromBody;
using people_perception::removeInsideCylinders;
using people_perception::toDetectorCloud;
using people_perception::yawOf;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;
using test::bytesOf;
using test::makeFastLioCloud;

namespace {

void expectSameLayout(const PointCloud2& out, const PointCloud2& in, size_t points) {
  EXPECT_EQ(out.header, in.header);
  EXPECT_EQ(out.fields, in.fields);
  EXPECT_EQ(out.point_step, in.point_step);
  EXPECT_EQ(out.is_bigendian, in.is_bigendian);
  EXPECT_EQ(out.is_dense, in.is_dense);
  EXPECT_EQ(out.height, 1u);
  EXPECT_EQ(out.width, points);
  EXPECT_EQ(out.row_step, points * in.point_step);
}

const Cylinder kPerson{1.0, 0.0, 0.4, -0.5, 1.5};
const Cylinder kOther{5.0, 5.0, 0.35, -0.5, 1.5};

TEST(RemoveInsideCylinders, TakesOutThePointsInsideAndKeepsTheOthersByteForByte) {
  const PointCloud2 in = makeFastLioCloud({
      {1.0F, 0.0F, 0.5F, 1.0F},    // 0 on the axis of the person
      {1.5F, 0.0F, 0.5F, 1.0F},    // 1 beside the person: 0.5 m from the axis, radius 0.4
      {1.0F, 0.1F, 1.6F, 1.0F},    // 2 above
      {1.0F, -0.1F, -0.6F, 1.0F},  // 3 below
      {1.39F, 0.0F, 0.0F, 1.0F},   // 4 inside, close to the surface
      {5.0F, 5.2F, 0.2F, 1.0F},    // 5 inside the other cylinder
      {1.0F, 0.41F, 0.0F, 1.0F},   // 6 just outside
  });
  PointCloud2 kept, removed;
  const auto count = removeInsideCylinders(in, {kPerson, kOther}, Eigen::Isometry3d::Identity(), kept, &removed);

  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 3u);
  expectSameLayout(kept, in, 4);
  EXPECT_EQ(kept.data, bytesOf(in, {1, 2, 3, 6}));
  expectSameLayout(removed, in, 3);
  EXPECT_EQ(removed.data, bytesOf(in, {0, 4, 5}));
}

TEST(RemoveInsideCylinders, WithoutCylindersTheCloudIsUnchanged) {
  const PointCloud2 in = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}, {2.0F, 0.0F, 0.5F, 1.0F}});
  PointCloud2 kept;
  const auto count = removeInsideCylinders(in, {}, Eigen::Isometry3d::Identity(), kept);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 0u);
  EXPECT_EQ(kept, in);
}

// The body cloud of SC-PGO: the cylinders stand vertical in the odometry frame, the cloud is in the frame of a robot
// that is somewhere else and pitched.
TEST(RemoveInsideCylinders, TestsACloudOfAnotherFrameThroughItsPose) {
  const Eigen::Isometry3d odom_from_body = Eigen::Translation3d(2.0, 1.0, 0.5) *
                                           Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()) *
                                           Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY());
  const Eigen::Vector3f on_person = (odom_from_body.inverse() * Eigen::Vector3d(1.0, 0.0, 0.5)).cast<float>();
  const PointCloud2 in = makeFastLioCloud({
      {on_person.x(), on_person.y(), on_person.z(), 1.0F},  // 0 on the axis of the person, seen from the body
      {1.0F, 0.0F, 0.5F, 1.0F},                             // 1 the same numbers in the body frame: elsewhere
  });
  PointCloud2 kept;
  const auto count = removeInsideCylinders(in, {kPerson}, odom_from_body, kept);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 1u);
  EXPECT_EQ(kept.data, bytesOf(in, {1}));
}

TEST(RemoveInsideCylinders, KeepsPointsThatAreNotFinite) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const PointCloud2 in = makeFastLioCloud({{nan, nan, nan, 0.0F}, {1.0F, 0.0F, 0.5F, 1.0F}});
  PointCloud2 kept;
  const auto count = removeInsideCylinders(in, {kPerson}, Eigen::Isometry3d::Identity(), kept);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 1u);
  EXPECT_EQ(kept.data, bytesOf(in, {0}));
}

TEST(RemoveInsideCylinders, OrganizedCloudComesOutUnorganized) {
  const PointCloud2 in = makeFastLioCloud(
      {{1.0F, 0.0F, 0.5F, 1.0F}, {2.0F, 0.0F, 0.5F, 1.0F}, {3.0F, 0.0F, 0.5F, 1.0F}, {4.0F, 0.0F, 0.5F, 1.0F}},
      "rslidar_lio", 10, 699999999, 2);
  PointCloud2 kept;
  ASSERT_TRUE(removeInsideCylinders(in, {kPerson}, Eigen::Isometry3d::Identity(), kept).has_value());
  expectSameLayout(kept, in, 3);
  EXPECT_EQ(kept.data, bytesOf(in, {1, 2, 3}));
}

TEST(RemoveInsideCylinders, RefusesACloudWithoutXyz) {
  PointCloud2 in = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}});
  in.fields.erase(in.fields.begin() + 1);  // no y
  PointCloud2 kept;
  EXPECT_FALSE(removeInsideCylinders(in, {kPerson}, Eigen::Isometry3d::Identity(), kept).has_value());
}

TEST(RemoveInsideCylinders, RefusesACloudWithLessDataThanItDeclares) {
  PointCloud2 in = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}, {2.0F, 0.0F, 0.5F, 1.0F}});
  in.data.resize(in.data.size() - 1);
  PointCloud2 kept;
  EXPECT_FALSE(removeInsideCylinders(in, {kPerson}, Eigen::Isometry3d::Identity(), kept).has_value());
}

// --------------------------------------------------------------------------------------------------------------------

const DetectorCloudParams kCrop{0.4, 21.0, -1.2, 2.2, 1.0};

struct Xyzirc {
  float x, y, z;
  uint8_t intensity, return_type;
  uint16_t channel;
};

Xyzirc detectorPoint(const PointCloud2& cloud, size_t i) {
  Xyzirc point{};
  const uint8_t* data = cloud.data.data() + i * cloud.point_step;
  std::memcpy(&point.x, data + 0, 4);
  std::memcpy(&point.y, data + 4, 4);
  std::memcpy(&point.z, data + 8, 4);
  point.intensity = data[12];
  point.return_type = data[13];
  std::memcpy(&point.channel, data + 14, 2);
  return point;
}

Eigen::Quaterniond bodyOrientation(double yaw, double pitch, double roll) {
  return Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
}

TEST(LevelFromBody, KeepsTheYawAndTakesOutRollAndPitch) {
  const Eigen::Quaterniond odom_from_body = bodyOrientation(0.7, 0.1, -0.05);
  EXPECT_NEAR(yawOf(odom_from_body), 0.7, 1e-9);

  // odom_from_level is the yaw alone
  const Eigen::Matrix3d odom_from_level = odom_from_body.toRotationMatrix() * levelFromBody(odom_from_body).transpose();
  EXPECT_TRUE(odom_from_level.isApprox(Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix(), 1e-9));
}

// 3 deg of pitch lift the floor 20 m ahead by a meter in the body frame; in the level frame it stays where it is.
TEST(LevelFromBody, FloorStaysLevelWhileTheBodyPitches) {
  const Eigen::Quaterniond odom_from_body = bodyOrientation(0.0, 3.0 * M_PI / 180.0, 0.0);
  const Eigen::Vector3d floor_ahead(20.0, 0.0, -0.47);  // odometry axes, from the sensor
  const Eigen::Vector3d in_body = odom_from_body.inverse() * floor_ahead;
  EXPECT_GT(std::abs(in_body.z() - floor_ahead.z()), 1.0);
  EXPECT_TRUE((levelFromBody(odom_from_body) * in_body).isApprox(floor_ahead, 1e-9));
}

TEST(ToDetectorCloud, HasThePointXyzircLayoutOfAutoware) {
  const PointCloud2 body = makeFastLioCloud({{2.0F, 0.0F, 0.0F, 1.0F}});
  PointCloud2 out;
  ASSERT_TRUE(toDetectorCloud(body, Eigen::Matrix3d::Identity(), kCrop, "sensor_level", out));

  ASSERT_EQ(out.fields.size(), 6u);
  const struct {
    const char* name;
    uint32_t offset;
    uint8_t datatype;
  } expected[] = {{"x", 0, PointField::FLOAT32},        {"y", 4, PointField::FLOAT32},
                  {"z", 8, PointField::FLOAT32},        {"intensity", 12, PointField::UINT8},
                  {"return_type", 13, PointField::UINT8}, {"channel", 14, PointField::UINT16}};
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(out.fields[i].name, expected[i].name);
    EXPECT_EQ(out.fields[i].offset, expected[i].offset);
    EXPECT_EQ(out.fields[i].datatype, expected[i].datatype);
    EXPECT_EQ(out.fields[i].count, 1u);
  }
  EXPECT_EQ(out.point_step, 16u);
  EXPECT_EQ(out.height, 1u);
  EXPECT_EQ(out.width, 1u);
  EXPECT_EQ(out.row_step, 16u);
  EXPECT_EQ(out.data.size(), 16u);
  EXPECT_FALSE(out.is_bigendian);
  EXPECT_EQ(out.header.frame_id, "sensor_level");
  EXPECT_EQ(out.header.stamp, body.header.stamp);
}

TEST(ToDetectorCloud, DropsTheRobotTheFarPointsAndWhatIsOutsideTheHeightWindow) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const PointCloud2 body = makeFastLioCloud({
      {2.0F, 0.0F, 0.0F, 100.4F},   // kept
      {0.2F, 0.1F, 0.0F, 1.0F},     // on the robot: 0.22 m from the sensor axis
      {0.1F, 0.1F, 1.5F, 1.0F},     // above the robot, still within the radius
      {25.0F, 0.0F, 0.0F, 1.0F},    // beyond 21 m
      {15.0F, 15.0F, 0.0F, 1.0F},   // 21.2 m away, inside the square
      {3.0F, 0.0F, 2.5F, 1.0F},     // above the window
      {3.0F, 0.0F, -1.3F, 1.0F},    // below the window
      {nan, 0.0F, 0.0F, 1.0F},      // no return
      {0.0F, 0.5F, 1.0F, 300.0F},   // kept, intensity saturates
      {-4.0F, -3.0F, -0.4F, -5.0F}, // kept, intensity floors at 0
  });
  PointCloud2 out;
  ASSERT_TRUE(toDetectorCloud(body, Eigen::Matrix3d::Identity(), kCrop, "sensor_level", out));

  ASSERT_EQ(out.width, 3u);
  ASSERT_EQ(out.data.size(), 3u * 16u);
  const Xyzirc a = detectorPoint(out, 0), b = detectorPoint(out, 1), c = detectorPoint(out, 2);
  EXPECT_EQ(a.x, 2.0F);
  EXPECT_EQ(a.y, 0.0F);
  EXPECT_EQ(a.z, 0.0F);
  EXPECT_EQ(a.intensity, 100);
  EXPECT_EQ(a.return_type, 0);
  EXPECT_EQ(a.channel, 0);
  EXPECT_EQ(b.y, 0.5F);
  EXPECT_EQ(b.intensity, 255);
  EXPECT_EQ(c.x, -4.0F);
  EXPECT_EQ(c.intensity, 0);
}

TEST(ToDetectorCloud, PutsThePointsInTheLevelFrame) {
  // Body yawed 0.6 and pitched 0.1: the level frame differs from the body by the pitch alone
  const Eigen::Quaterniond odom_from_body = bodyOrientation(0.6, 0.1, 0.0);
  const PointCloud2 body = makeFastLioCloud({{5.0F, 1.0F, 0.2F, 1.0F}});
  PointCloud2 out;
  ASSERT_TRUE(toDetectorCloud(body, levelFromBody(odom_from_body), kCrop, "sensor_level", out));

  ASSERT_EQ(out.width, 1u);
  const Xyzirc point = detectorPoint(out, 0);
  // Ry(0.1) * (5, 1, 0.2): x = 5 cos 0.1 + 0.2 sin 0.1, z = -5 sin 0.1 + 0.2 cos 0.1
  EXPECT_NEAR(point.x, 4.99499, 1e-4);
  EXPECT_NEAR(point.y, 1.0, 1e-5);
  EXPECT_NEAR(point.z, -0.30017, 1e-4);
}

TEST(ToDetectorCloud, EmptyCloudGivesAValidEmptyCloud) {
  const PointCloud2 body = makeFastLioCloud({});
  PointCloud2 out;
  ASSERT_TRUE(toDetectorCloud(body, Eigen::Matrix3d::Identity(), kCrop, "sensor_level", out));
  EXPECT_EQ(out.width, 0u);
  EXPECT_EQ(out.height, 1u);
  EXPECT_TRUE(out.data.empty());
  EXPECT_EQ(out.point_step, 16u);
  EXPECT_EQ(out.fields.size(), 6u);
  EXPECT_EQ(out.header.stamp, body.header.stamp);
}

TEST(ToDetectorCloud, CloudWithoutIntensityGetsZero) {
  PointCloud2 body = makeFastLioCloud({{2.0F, 0.0F, 0.0F, 77.0F}});
  body.fields.erase(body.fields.begin() + 6);  // no intensity
  PointCloud2 out;
  ASSERT_TRUE(toDetectorCloud(body, Eigen::Matrix3d::Identity(), kCrop, "sensor_level", out));
  ASSERT_EQ(out.width, 1u);
  EXPECT_EQ(detectorPoint(out, 0).intensity, 0);
}

TEST(ToDetectorCloud, RefusesACloudWithoutXyz) {
  PointCloud2 body = makeFastLioCloud({{2.0F, 0.0F, 0.0F, 1.0F}});
  body.fields.erase(body.fields.begin());  // no x
  PointCloud2 out;
  EXPECT_FALSE(toDetectorCloud(body, Eigen::Matrix3d::Identity(), kCrop, "sensor_level", out));
}

}  // namespace
