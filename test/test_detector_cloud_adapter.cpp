#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include <Eigen/Geometry>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "people_perception/DetectorCloudAdapter.h"

#include "RosTest.h"
#include "TestClouds.h"

using nav_msgs::msg::Odometry;
using people_perception::DetectorCloudAdapter;
using sensor_msgs::msg::PointCloud2;
using tf2_msgs::msg::TFMessage;
using test::makeFastLioCloud;
using test::makeOdometry;
using namespace std::chrono_literals;

namespace {

float floatAt(const PointCloud2& cloud, size_t point, size_t offset) {
  float value;
  std::memcpy(&value, cloud.data.data() + point * cloud.point_step + offset, sizeof(float));
  return value;
}

class DetectorCloudAdapterTest : public test::RosTest {
 protected:
  void SetUp() override {
    RosTest::SetUp();
    adapter_ = std::make_shared<DetectorCloudAdapter>(test::shippedConfig("detector_cloud_adapter.yaml"));
    executor_.add_node(adapter_);
    cloud_ = io_->create_publisher<PointCloud2>("/cloud_registered_body", 20);
    odometry_ = io_->create_publisher<Odometry>("/Odometry", 20);
    out_ = collect<PointCloud2>("/perception/detector/pointcloud");
    tf_ = collect<TFMessage>("/tf");
    ASSERT_TRUE(spinUntil([&] {
      return cloud_->get_subscription_count() == 1 && odometry_->get_subscription_count() == 1 &&
             io_->count_publishers("/perception/detector/pointcloud") == 1 && io_->count_publishers("/tf") == 1;
    })) << "the adapter does not read /cloud_registered_body and /Odometry, or does not publish its cloud and TF";
  }

  std::shared_ptr<DetectorCloudAdapter> adapter_;
  rclcpp::Publisher<PointCloud2>::SharedPtr cloud_;
  rclcpp::Publisher<Odometry>::SharedPtr odometry_;
  std::shared_ptr<std::vector<PointCloud2>> out_;
  std::shared_ptr<std::vector<TFMessage>> tf_;
};

TEST_F(DetectorCloudAdapterTest, ScanComesOutLevelledWithTheTransformOfItsStamp) {
  // Body 0.6 rad of yaw and 0.1 of pitch: the level frame differs from the body by the pitch alone
  const Eigen::Quaterniond orientation(Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ()) *
                                       Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()));
  odometry_->publish(makeOdometry(10, 699999999, Eigen::Vector3d(1.0, 2.0, 0.5), orientation));
  cloud_->publish(makeFastLioCloud({{5.0F, 1.0F, 0.2F, 1.0F}}));

  ASSERT_TRUE(spinUntil([&] { return !out_->empty() && !tf_->empty(); }));
  const PointCloud2& cloud = out_->front();
  EXPECT_EQ(cloud.header.frame_id, "sensor_level");
  EXPECT_EQ(cloud.header.stamp.sec, 10);
  EXPECT_EQ(cloud.header.stamp.nanosec, 699999999u);
  EXPECT_EQ(cloud.point_step, 16u);
  ASSERT_EQ(cloud.width, 1u);
  // Ry(0.1) * (5, 1, 0.2)
  EXPECT_NEAR(floatAt(cloud, 0, 0), 4.99499, 1e-4);
  EXPECT_NEAR(floatAt(cloud, 0, 4), 1.0, 1e-5);
  EXPECT_NEAR(floatAt(cloud, 0, 8), -0.30017, 1e-4);

  ASSERT_EQ(tf_->front().transforms.size(), 1u);
  const auto& transform = tf_->front().transforms.front();
  EXPECT_EQ(transform.header.frame_id, "camera_init");
  EXPECT_EQ(transform.child_frame_id, "sensor_level");
  EXPECT_EQ(transform.header.stamp, cloud.header.stamp);
  EXPECT_DOUBLE_EQ(transform.transform.translation.x, 1.0);
  EXPECT_DOUBLE_EQ(transform.transform.translation.y, 2.0);
  EXPECT_DOUBLE_EQ(transform.transform.translation.z, 0.5);
  // The yaw alone: (0, 0, sin 0.3, cos 0.3)
  EXPECT_NEAR(transform.transform.rotation.x, 0.0, 1e-9);
  EXPECT_NEAR(transform.transform.rotation.y, 0.0, 1e-9);
  EXPECT_NEAR(transform.transform.rotation.z, 0.295520, 1e-6);
  EXPECT_NEAR(transform.transform.rotation.w, 0.955336, 1e-6);
}

TEST_F(DetectorCloudAdapterTest, OdometryArrivingAfterItsCloudStillPairs) {
  cloud_->publish(makeFastLioCloud({{5.0F, 1.0F, 0.2F, 1.0F}}));
  spinFor(100ms);
  EXPECT_TRUE(out_->empty());
  odometry_->publish(makeOdometry(10, 699999999));
  ASSERT_TRUE(spinUntil([&] { return !out_->empty(); }));
  EXPECT_EQ(out_->front().header.stamp.nanosec, 699999999u);
}

// On a loaded machine the small odometry message can reach this node after the next 1 MB cloud (seen replaying at
// 10 Hz on 2026-10-05): the scan must wait for it, not be dropped when the next cloud arrives.
TEST_F(DetectorCloudAdapterTest, OdometryLateByMoreThanAScanStillPairs) {
  cloud_->publish(makeFastLioCloud({{5.0F, 1.0F, 0.2F, 1.0F}}, "rslidar_lio", 10, 0));
  cloud_->publish(makeFastLioCloud({{6.0F, 1.0F, 0.2F, 1.0F}}, "rslidar_lio", 10, 100000000));
  spinFor(200ms);
  EXPECT_TRUE(out_->empty());
  odometry_->publish(makeOdometry(10, 0));
  odometry_->publish(makeOdometry(10, 100000000));
  ASSERT_TRUE(spinUntil([&] { return out_->size() == 2; }));
  EXPECT_EQ((*out_)[0].header.stamp.nanosec, 0u);
  EXPECT_EQ((*out_)[1].header.stamp.nanosec, 100000000u);
}

// Odometry stopped: the clouds must not pile up for ever.
TEST_F(DetectorCloudAdapterTest, CloudsWaitingForOdometryAreBounded) {
  for (int32_t i = 0; i < 30; ++i) {
    cloud_->publish(makeFastLioCloud({{5.0F, 1.0F, 0.2F, 1.0F}}, "rslidar_lio", 50 + i, 0));
    spinFor(20ms);
  }
  // Only the last cloud_queue (10) scans can still be paired
  for (int32_t i = 0; i < 30; ++i) {
    odometry_->publish(makeOdometry(50 + i, 0));
  }
  ASSERT_TRUE(spinUntil([&] { return out_->size() == 10; }));
  spinFor(200ms);
  ASSERT_EQ(out_->size(), 10u);
  EXPECT_EQ(out_->front().header.stamp.sec, 70);
  EXPECT_EQ(out_->back().header.stamp.sec, 79);
}

TEST_F(DetectorCloudAdapterTest, ScanWhoseOdometryNeverComesIsDropped) {
  cloud_->publish(makeFastLioCloud({{5.0F, 1.0F, 0.2F, 1.0F}}, "rslidar_lio", 10, 0));
  spinFor(100ms);
  odometry_->publish(makeOdometry(11, 0));
  cloud_->publish(makeFastLioCloud({{6.0F, 1.0F, 0.2F, 1.0F}}, "rslidar_lio", 11, 0));
  ASSERT_TRUE(spinUntil([&] { return !out_->empty(); }));
  spinFor(200ms);
  ASSERT_EQ(out_->size(), 1u);
  EXPECT_EQ(out_->front().header.stamp.sec, 11);
}

// Every scan gets its cloud, an empty one too: the detector answers each cloud, and the filter waits for that answer.
TEST_F(DetectorCloudAdapterTest, EmptyScanStillGivesACloud) {
  odometry_->publish(makeOdometry(10, 699999999));
  cloud_->publish(makeFastLioCloud({}));
  ASSERT_TRUE(spinUntil([&] { return !out_->empty(); }));
  EXPECT_EQ(out_->front().width, 0u);
  EXPECT_EQ(out_->front().point_step, 16u);
  EXPECT_EQ(out_->front().header.frame_id, "sensor_level");
}

TEST_F(DetectorCloudAdapterTest, EveryScanGivesOneCloudInOrder) {
  const int32_t scans = 15;
  for (int32_t i = 0; i < scans; ++i) {
    odometry_->publish(makeOdometry(i, 0));
    cloud_->publish(makeFastLioCloud({{5.0F, 1.0F, 0.2F, 1.0F}}, "rslidar_lio", i, 0));
    ASSERT_TRUE(spinUntil([&] { return out_->size() > static_cast<size_t>(i); })) << "scan " << i << " lost";
  }
  spinFor(200ms);
  ASSERT_EQ(out_->size(), static_cast<size_t>(scans));
  ASSERT_EQ(tf_->size(), static_cast<size_t>(scans));
  for (int32_t i = 0; i < scans; ++i) {
    EXPECT_EQ((*out_)[static_cast<size_t>(i)].header.stamp.sec, i);
    EXPECT_EQ((*tf_)[static_cast<size_t>(i)].transforms.front().header.stamp.sec, i);
  }
}

}  // namespace
