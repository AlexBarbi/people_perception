#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "people_perception/DynamicPointsFilter.h"

#include "RosTest.h"
#include "TestClouds.h"

using autoware_perception_msgs::msg::TrackedObject;
using autoware_perception_msgs::msg::TrackedObjects;
using diagnostic_msgs::msg::DiagnosticArray;
using diagnostic_msgs::msg::DiagnosticStatus;
using nav_msgs::msg::Odometry;
using people_perception::DynamicPointsFilter;
using sensor_msgs::msg::PointCloud2;
using test::bytesOf;
using test::makeFastLioCloud;
using test::makeOdometry;
using test::Xyzi;
using namespace std::chrono_literals;

namespace {

// The stamp of every scan below unless said otherwise.
const int32_t kSec = 10;
const uint32_t kNanosec = 699999999;

std::vector<Xyzi> manyPoints(size_t count) {
  std::vector<Xyzi> points;
  for (size_t i = 0; i < count; ++i) {
    points.push_back({0.01F * static_cast<float>(i), 5.0F - 0.003F * static_cast<float>(i), 0.3F, 7.0F});
  }
  return points;
}

// A person as people_tracker reports them: a 0.5 x 0.3 x 1.7 m box whose centre is 0.4 m above the frame origin, so
// with the margins that ship a cylinder of radius 0.45 m from z = -0.55 to 1.45 m.
TrackedObject makeTrack(double x, double y, double vx = 0.0, double vy = 0.0, double length = 0.5, double width = 0.3,
                        double position_variance = 0.0) {
  TrackedObject track;
  track.kinematics.pose_with_covariance.pose.position.x = x;
  track.kinematics.pose_with_covariance.pose.position.y = y;
  track.kinematics.pose_with_covariance.pose.position.z = 0.4;
  track.kinematics.pose_with_covariance.pose.orientation.w = 1.0;
  track.kinematics.pose_with_covariance.covariance[0] = position_variance;
  track.kinematics.pose_with_covariance.covariance[7] = position_variance;
  track.kinematics.twist_with_covariance.twist.linear.x = vx;
  track.kinematics.twist_with_covariance.twist.linear.y = vy;
  track.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  track.shape.dimensions.x = length;
  track.shape.dimensions.y = width;
  track.shape.dimensions.z = 1.7;
  return track;
}

TrackedObjects makeTracks(const std::vector<TrackedObject>& objects, int32_t sec = kSec, uint32_t nanosec = kNanosec,
                          const std::string& frame = "camera_init") {
  TrackedObjects tracks;
  tracks.header.frame_id = frame;
  tracks.header.stamp.sec = sec;
  tracks.header.stamp.nanosec = nanosec;
  tracks.objects = objects;
  return tracks;
}

class DynamicPointsFilterTest : public test::RosTest {
 protected:
  // The yaml that ships: a pass-through.
  static rclcpp::NodeOptions passThrough() {
    return test::shippedConfig("dynamic_points_filter.yaml")
        .append_parameter_override("duplicate_check_wait", 0.2);  // the shipped wait would add seconds to every test
  }

  // The same with the filtering on, and a max_wait the tests can tell from "at once" and from "never".
  static rclcpp::NodeOptions filtering(double max_wait = 5.0) {
    return passThrough().append_parameter_override("enabled", true).append_parameter_override("max_wait", max_wait);
  }

  void start(const rclcpp::NodeOptions& options, bool filters) {
    filter_ = std::make_shared<DynamicPointsFilter>(options);
    executor_.add_node(filter_);
    registered_ = io_->create_publisher<PointCloud2>("/cloud_registered", 20);
    body_ = io_->create_publisher<PointCloud2>("/cloud_registered_body", 20);
    odometry_ = io_->create_publisher<Odometry>("/Odometry", 20);
    tracks_ = io_->create_publisher<TrackedObjects>("/perception/tracks_all", 20);
    registered_static_ = collect<PointCloud2>("/registered_scan_static");
    body_static_ = collect<PointCloud2>("/cloud_registered_body_static");
    removed_ = collect<PointCloud2>("/perception/dynamic_points");
    ASSERT_TRUE(spinUntil([&] {
      return registered_->get_subscription_count() == 1 && body_->get_subscription_count() == 1 &&
             io_->count_publishers("/registered_scan_static") == 1 &&
             io_->count_publishers("/cloud_registered_body_static") == 1 &&
             (!filters || (tracks_->get_subscription_count() == 1 && odometry_->get_subscription_count() == 1 &&
                           io_->count_publishers("/perception/dynamic_points") == 1));
    })) << "the filter did not connect its topics";
  }

  std::shared_ptr<DynamicPointsFilter> filter_;
  rclcpp::Publisher<PointCloud2>::SharedPtr registered_;
  rclcpp::Publisher<PointCloud2>::SharedPtr body_;
  rclcpp::Publisher<Odometry>::SharedPtr odometry_;
  rclcpp::Publisher<TrackedObjects>::SharedPtr tracks_;
  std::shared_ptr<std::vector<PointCloud2>> registered_static_;
  std::shared_ptr<std::vector<PointCloud2>> body_static_;
  std::shared_ptr<std::vector<PointCloud2>> removed_;
};

// ---------------------------------------------------------------------------------------------- enabled: false

TEST_F(DynamicPointsFilterTest, PassThroughGivesTareTheRegisteredCloudUnchanged) {
  start(passThrough(), false);
  const PointCloud2 sent = makeFastLioCloud(manyPoints(2000), "camera_init");
  registered_->publish(sent);
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }));
  EXPECT_EQ(registered_static_->front(), sent);
}

TEST_F(DynamicPointsFilterTest, PassThroughGivesScPgoTheBodyCloudUnchanged) {
  start(passThrough(), false);
  const PointCloud2 sent = makeFastLioCloud(manyPoints(2000), "rslidar_lio");
  body_->publish(sent);
  ASSERT_TRUE(spinUntil([&] { return !body_static_->empty(); }));
  EXPECT_EQ(body_static_->front(), sent);
}

TEST_F(DynamicPointsFilterTest, PassThroughForwardsAnEmptyCloud) {
  start(passThrough(), false);
  const PointCloud2 sent = makeFastLioCloud({}, "camera_init");
  registered_->publish(sent);
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }));
  EXPECT_EQ(registered_static_->front(), sent);
}

TEST_F(DynamicPointsFilterTest, PassThroughForwardsEveryScanOnceInOrder) {
  start(passThrough(), false);
  const int32_t scans = 20;
  for (int32_t i = 0; i < scans; ++i) {
    body_->publish(makeFastLioCloud(manyPoints(100), "rslidar_lio", i, 0));
    ASSERT_TRUE(spinUntil([&] { return body_static_->size() > static_cast<size_t>(i); })) << "scan " << i << " lost";
  }
  spinFor(200ms);  // a duplicate would arrive now
  ASSERT_EQ(body_static_->size(), static_cast<size_t>(scans));
  for (int32_t i = 0; i < scans; ++i) {
    EXPECT_EQ((*body_static_)[static_cast<size_t>(i)].header.stamp.sec, i);
  }
}

TEST_F(DynamicPointsFilterTest, TopicsComeFromTheParameters) {
  filter_ = std::make_shared<DynamicPointsFilter>(
      passThrough()
          .append_parameter_override("registered_cloud.input_topic", "/other_scan")
          .append_parameter_override("registered_cloud.output_topic", "/other_scan_static"));
  executor_.add_node(filter_);
  const auto input = io_->create_publisher<PointCloud2>("/other_scan", 20);
  const auto output = collect<PointCloud2>("/other_scan_static");
  ASSERT_TRUE(spinUntil(
      [&] { return input->get_subscription_count() == 1 && io_->count_publishers("/other_scan_static") == 1; }));

  const PointCloud2 sent = makeFastLioCloud(manyPoints(10), "camera_init");
  input->publish(sent);
  ASSERT_TRUE(spinUntil([&] { return !output->empty(); }));
  EXPECT_EQ(output->front(), sent);
}

// Best effort loses the 1 MB clouds of FAST-LIO to fragment drops, see common.lid_reliable in fast_lio_airy.yaml.
TEST_F(DynamicPointsFilterTest, InputsAreSubscribedReliable) {
  start(passThrough(), false);
  for (const char* topic : {"/cloud_registered", "/cloud_registered_body"}) {
    const auto subscriptions = io_->get_subscriptions_info_by_topic(topic);
    ASSERT_EQ(subscriptions.size(), 1u);
    EXPECT_EQ(subscriptions.front().qos_profile().reliability(), rclcpp::ReliabilityPolicy::Reliable) << topic;
  }
}

// Two filters hand every scan to SC-PGO twice: it pairs the second copy with the odometry of the next scan and falls
// one scan further behind on every scan. Seen on 2026-10-05 with the launch started by hand next to
// fast_lio_slam_launch.xml.
TEST_F(DynamicPointsFilterTest, SecondFilterOnTheSameTopicsRefusesToStart) {
  start(passThrough(), false);

  auto second = std::make_shared<DynamicPointsFilter>(passThrough());
  executor_.add_node(second);
  EXPECT_THROW(spinFor(1s), std::runtime_error);
  executor_.remove_node(second);
  second.reset();

  body_->publish(makeFastLioCloud(manyPoints(100), "rslidar_lio"));
  ASSERT_TRUE(spinUntil([&] { return !body_static_->empty(); }));
  spinFor(300ms);  // the copy of the second filter would arrive now
  EXPECT_EQ(body_static_->size(), 1u);
}

// ----------------------------------------------------------------------------------------------- enabled: true

TEST_F(DynamicPointsFilterTest, CloudWaitsForTheTracksOfItsScanAndComesOutWithoutThePeople) {
  start(filtering(), true);
  const PointCloud2 cloud = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F},   // 0 on the person
                                              {3.0F, 0.0F, 0.5F, 1.0F},   // 1 a wall
                                              {1.1F, 0.1F, 1.0F, 1.0F},   // 2 on the person
                                              {1.0F, 1.0F, 0.5F, 1.0F}},  // 3 a meter beside the person
                                             "camera_init");
  registered_->publish(cloud);
  spinFor(300ms);
  EXPECT_TRUE(registered_static_->empty()) << "the cloud went out before the tracker answered for its scan";

  tracks_->publish(makeTracks({makeTrack(1.0, 0.0)}));
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty() && !removed_->empty(); }, 2s));
  const PointCloud2& out = registered_static_->front();
  EXPECT_EQ(out.header, cloud.header);
  EXPECT_EQ(out.fields, cloud.fields);
  EXPECT_EQ(out.point_step, cloud.point_step);
  EXPECT_EQ(out.width, 2u);
  EXPECT_EQ(out.row_step, 2u * cloud.point_step);
  EXPECT_EQ(out.data, bytesOf(cloud, {1, 3}));
  // For diagnostics: what was taken out, same frame and stamp
  EXPECT_EQ(removed_->front().header, cloud.header);
  EXPECT_EQ(removed_->front().data, bytesOf(cloud, {0, 2}));
}

TEST_F(DynamicPointsFilterTest, TracksThatCameBeforeTheCloudAreUsedWhenItArrives) {
  start(filtering(), true);
  tracks_->publish(makeTracks({makeTrack(1.0, 0.0)}));
  spinFor(200ms);
  const PointCloud2 cloud = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}, {3.0F, 0.0F, 0.5F, 1.0F}}, "camera_init");
  registered_->publish(cloud);
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }, 2s));
  EXPECT_EQ(registered_static_->front().data, bytesOf(cloud, {1}));
}

TEST_F(DynamicPointsFilterTest, SceneWithoutPeopleGoesOutUnchangedAtOnce) {
  start(filtering(), true);
  const PointCloud2 cloud = makeFastLioCloud(manyPoints(500), "camera_init");
  registered_->publish(cloud);
  tracks_->publish(makeTracks({}));
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty() && !removed_->empty(); }, 2s));
  EXPECT_EQ(registered_static_->front(), cloud);
  EXPECT_EQ(removed_->front().width, 0u);
}

// SC-PGO keeps its scans in the frame of the robot, which is elsewhere and pitched: the people are found through the
// odometry of the scan.
TEST_F(DynamicPointsFilterTest, BodyCloudLosesThePeopleThroughTheOdometryOfItsScan) {
  start(filtering(), true);
  const Eigen::Vector3d position(2.0, 1.0, 0.5);
  const Eigen::Quaterniond orientation(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()) *
                                       Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
  const Eigen::Vector3f on_person = (orientation.inverse() * (Eigen::Vector3d(1.0, 0.0, 0.5) - position)).cast<float>();
  const PointCloud2 cloud = makeFastLioCloud({{on_person.x(), on_person.y(), on_person.z(), 1.0F},  // 0 on the person
                                              {1.0F, 0.0F, 0.5F, 1.0F}},  // 1 the same numbers in the body frame
                                             "rslidar_lio");
  odometry_->publish(makeOdometry(kSec, kNanosec, position, orientation));
  body_->publish(cloud);
  tracks_->publish(makeTracks({makeTrack(1.0, 0.0)}));
  ASSERT_TRUE(spinUntil([&] { return !body_static_->empty(); }, 2s));
  EXPECT_EQ(body_static_->front().header, cloud.header);
  EXPECT_EQ(body_static_->front().data, bytesOf(cloud, {1}));
}

TEST_F(DynamicPointsFilterTest, BodyCloudWithoutItsOdometryGoesOutUnfilteredAfterMaxWait) {
  start(filtering(0.3), true);
  const PointCloud2 cloud = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}}, "rslidar_lio");
  body_->publish(cloud);
  tracks_->publish(makeTracks({makeTrack(1.0, 0.0)}));
  spinFor(150ms);
  EXPECT_TRUE(body_static_->empty());
  ASSERT_TRUE(spinUntil([&] { return !body_static_->empty(); }, 2s));
  EXPECT_EQ(body_static_->front(), cloud);
}

// On a loaded machine the odometry of a scan can come after the tracks of the next one (seen replaying at 10 Hz on
// 2026-10-05, a tenth of the body clouds went out unfiltered): the cloud must wait for it.
TEST_F(DynamicPointsFilterTest, BodyCloudWaitsForALateOdometryEvenWhenLaterTracksAreHere) {
  start(filtering(), true);
  const PointCloud2 cloud = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}, {3.0F, 0.0F, 0.5F, 1.0F}}, "rslidar_lio", 10, 0);
  body_->publish(cloud);
  tracks_->publish(makeTracks({makeTrack(1.0, 0.0)}, 10, 0));
  spinFor(100ms);
  tracks_->publish(makeTracks({}, 10, 100000000));
  spinFor(200ms);
  EXPECT_TRUE(body_static_->empty()) << "the cloud went out before its odometry came";
  odometry_->publish(makeOdometry(10, 0));
  ASSERT_TRUE(spinUntil([&] { return !body_static_->empty(); }, 2s));
  EXPECT_EQ(body_static_->front().data, bytesOf(cloud, {1}));
}

// SC-PGO takes the clouds in the order they come and pairs each with the next odometry in its queue: a cloud that
// overtakes an earlier one shifts every later pair by a scan, for good.
TEST_F(DynamicPointsFilterTest, CloudNeverOvertakesAnEarlierOneThatStillWaits) {
  start(filtering(), true);
  const PointCloud2 first = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}, {3.0F, 0.0F, 0.5F, 1.0F}}, "rslidar_lio", 10, 0);
  const PointCloud2 second = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}}, "rslidar_lio", 10, 100000000);
  body_->publish(first);
  tracks_->publish(makeTracks({makeTrack(1.0, 0.0)}, 10, 0));
  spinFor(100ms);
  // The second scan has all it needs, the first still waits for its odometry
  odometry_->publish(makeOdometry(10, 100000000));
  tracks_->publish(makeTracks({}, 10, 100000000));
  body_->publish(second);
  spinFor(300ms);
  EXPECT_TRUE(body_static_->empty());
  odometry_->publish(makeOdometry(10, 0));
  ASSERT_TRUE(spinUntil([&] { return body_static_->size() == 2; }, 2s));
  EXPECT_EQ((*body_static_)[0].header.stamp.nanosec, 0u);
  EXPECT_EQ((*body_static_)[0].data, bytesOf(first, {1}));
  EXPECT_EQ((*body_static_)[1], second);
}

TEST_F(DynamicPointsFilterTest, CylinderIsTheBoxPlusTheMargins) {
  start(filtering(), true);
  const PointCloud2 cloud = makeFastLioCloud({{0.44F, 0.0F, 0.5F, 1.0F},    // 0 inside the 0.45 m radius
                                              {0.46F, 0.0F, 0.5F, 1.0F},    // 1 outside
                                              {0.0F, 0.1F, -0.54F, 1.0F},   // 2 above the bottom at -0.55 m
                                              {0.0F, 0.1F, -0.56F, 1.0F},   // 3 below it
                                              {0.1F, 0.0F, 1.44F, 1.0F},    // 4 below the top at 1.45 m
                                              {0.1F, 0.0F, 1.46F, 1.0F},    // 5 above it
                                              {5.34F, 5.0F, 0.5F, 1.0F},    // 6 thin box: inside min_radius 0.35 m
                                              {5.36F, 5.0F, 0.5F, 1.0F}},   // 7 outside
                                             "camera_init");
  registered_->publish(cloud);
  tracks_->publish(makeTracks({makeTrack(0.0, 0.0), makeTrack(5.0, 5.0, 0.0, 0.0, 0.1, 0.1)}));
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }, 2s));
  EXPECT_EQ(registered_static_->front().data, bytesOf(cloud, {1, 3, 5, 7}));
}

// A coasting track is somewhere around its prediction: people_tracker says how far in the position covariance.
TEST_F(DynamicPointsFilterTest, UncertainTrackGetsAWiderCylinder) {
  start(filtering(), true);
  const PointCloud2 cloud = makeFastLioCloud({{0.9F, 0.0F, 0.5F, 1.0F},   // 0 inside 0.45 + 0.5 m
                                              {1.0F, 0.0F, 0.5F, 1.0F}},  // 1 outside
                                             "camera_init");
  registered_->publish(cloud);
  tracks_->publish(makeTracks({makeTrack(0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.25)}));  // std 0.5 m
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }, 2s));
  EXPECT_EQ(registered_static_->front().data, bytesOf(cloud, {1}));
}

TEST_F(DynamicPointsFilterTest, CloudsComeOutInTheOrderOfTheScans) {
  start(filtering(), true);
  registered_->publish(makeFastLioCloud(manyPoints(10), "camera_init", 20, 0));
  registered_->publish(makeFastLioCloud(manyPoints(10), "camera_init", 20, 100000000));
  spinFor(200ms);
  tracks_->publish(makeTracks({}, 20, 0));
  tracks_->publish(makeTracks({}, 20, 100000000));
  ASSERT_TRUE(spinUntil([&] { return registered_static_->size() == 2; }, 2s));
  EXPECT_EQ((*registered_static_)[0].header.stamp.nanosec, 0u);
  EXPECT_EQ((*registered_static_)[1].header.stamp.nanosec, 100000000u);
}

// --------------------------------------------------------------------------------------- the detector is late or down

// The filter is not what avoids collisions: when the detector fails, the map still has to get its scans.
TEST_F(DynamicPointsFilterTest, WithoutAnyTracksTheCloudGoesOutUnchangedAfterMaxWait) {
  start(filtering(0.3), true);
  const PointCloud2 cloud = makeFastLioCloud(manyPoints(100), "camera_init");
  registered_->publish(cloud);
  spinFor(150ms);
  EXPECT_TRUE(registered_static_->empty());
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }, 2s));
  EXPECT_EQ(registered_static_->front(), cloud);
}

TEST_F(DynamicPointsFilterTest, AfterMaxWaitTheLastTracksAreMovedOnWithTheirVelocity) {
  start(filtering(0.3), true);
  tracks_->publish(makeTracks({makeTrack(1.0, 0.0, 1.0, 0.0)}, 10, 0));  // at x = 1 m, walking at 1 m/s along x
  spinFor(100ms);
  const PointCloud2 cloud = makeFastLioCloud({{1.5F, 0.0F, 0.5F, 1.0F},   // 0 where the person is half a second later
                                              {1.0F, 0.0F, 0.5F, 1.0F}},  // 1 where the person was
                                             "camera_init", 10, 500000000);
  registered_->publish(cloud);
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }, 2s));
  EXPECT_EQ(registered_static_->front().data, bytesOf(cloud, {1}));
}

TEST_F(DynamicPointsFilterTest, TracksOlderThanMaxTrackStalenessAreNotUsed) {
  start(filtering(0.3), true);
  tracks_->publish(makeTracks({makeTrack(1.0, 0.0)}, 10, 0));
  spinFor(100ms);
  const PointCloud2 cloud = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}}, "camera_init", 11, 500000000);  // 1.5 s later
  registered_->publish(cloud);
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }, 2s));
  EXPECT_EQ(registered_static_->front(), cloud);
}

TEST_F(DynamicPointsFilterTest, OnTimeoutPassthroughIgnoresTheLastTracks) {
  start(filtering(0.3).append_parameter_override("on_timeout", "passthrough"), true);
  tracks_->publish(makeTracks({makeTrack(1.0, 0.0)}, 10, 0));
  spinFor(100ms);
  const PointCloud2 cloud = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}}, "camera_init", 10, 100000000);
  registered_->publish(cloud);
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }, 2s));
  EXPECT_EQ(registered_static_->front(), cloud);
}

TEST_F(DynamicPointsFilterTest, OnTimeoutDropDiscardsTheCloud) {
  start(filtering(0.3).append_parameter_override("on_timeout", "drop"), true);
  registered_->publish(makeFastLioCloud(manyPoints(10), "camera_init"));
  spinFor(900ms);
  EXPECT_TRUE(registered_static_->empty());
}

// The tracker answers the scans in order: once it has answered a later one, an earlier one will never get its tracks.
TEST_F(DynamicPointsFilterTest, TracksOfALaterScanCloseTheEarlierCloudsAtOnce) {
  start(filtering(), true);  // max_wait 5 s: only the later tracks can let the cloud out within the 2 s below
  const PointCloud2 cloud = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}, {3.0F, 0.0F, 0.5F, 1.0F}}, "camera_init", 10, 0);
  registered_->publish(cloud);
  spinFor(200ms);
  tracks_->publish(makeTracks({makeTrack(1.0, 0.0)}, 10, 100000000));
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }, 2s));
  EXPECT_EQ(registered_static_->front().header, cloud.header);
  EXPECT_EQ(registered_static_->front().data, bytesOf(cloud, {1}));
}

TEST_F(DynamicPointsFilterTest, FullQueueLetsTheOldestCloudOut) {
  start(filtering().append_parameter_override("queue_size", 2), true);
  for (uint32_t i = 0; i < 3; ++i) {
    registered_->publish(makeFastLioCloud(manyPoints(10), "camera_init", 30, i * 100000000u));
  }
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }, 2s));
  spinFor(300ms);
  ASSERT_EQ(registered_static_->size(), 1u);
  EXPECT_EQ(registered_static_->front().header.stamp.nanosec, 0u);
}

TEST_F(DynamicPointsFilterTest, TracksOfAnotherFrameAreNotApplied) {
  start(filtering(), true);
  const PointCloud2 cloud = makeFastLioCloud({{1.0F, 0.0F, 0.5F, 1.0F}}, "camera_init");
  registered_->publish(cloud);
  tracks_->publish(makeTracks({makeTrack(1.0, 0.0)}, kSec, kNanosec, "aft_pgo"));
  ASSERT_TRUE(spinUntil([&] { return !registered_static_->empty(); }, 2s));
  EXPECT_EQ(registered_static_->front(), cloud);
}

TEST_F(DynamicPointsFilterTest, ScansWithoutTracksInTimeShowUpInTheDiagnostics) {
  const auto diagnostics = collect<DiagnosticArray>("/diagnostics");
  start(filtering(0.2), true);
  registered_->publish(makeFastLioCloud(manyPoints(10), "camera_init"));
  const auto warned = [&] {
    for (const DiagnosticArray& array : *diagnostics) {
      for (const DiagnosticStatus& status : array.status) {
        if (status.name.find("dynamic_points_filter") != std::string::npos && status.level == DiagnosticStatus::WARN) {
          return true;
        }
      }
    }
    return false;
  };
  EXPECT_TRUE(spinUntil(warned, 4s));
}

}  // namespace
