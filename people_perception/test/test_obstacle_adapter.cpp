#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <people_perception_msgs/msg/dynamic_obstacle_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "people_perception/ObstacleAdapter.h"

#include "RosTest.h"
#include "TestClouds.h"

using autoware_perception_msgs::msg::ObjectClassification;
using autoware_perception_msgs::msg::TrackedObject;
using autoware_perception_msgs::msg::TrackedObjects;
using nav_msgs::msg::Odometry;
using people_perception::ObstacleAdapter;
using people_perception::ObstacleParams;
using people_perception::toObstacles;
using people_perception_msgs::msg::DynamicObstacle;
using people_perception_msgs::msg::DynamicObstacleArray;
using test::makeOdometry;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;
using namespace std::chrono_literals;

namespace {

// The values of config/obstacle_adapter.yaml.
const ObstacleParams kShipped{10.0, 4, 0.30, 0.50, 0.10, 1.5};

// A confirmed person as people_tracker reports them; `tag` ends up in the first byte of the id.
TrackedObject makeTrack(double x, double y, uint8_t tag = 1, double length = 0.5, double width = 0.3,
                        double vx = 0.0, double vy = 0.0, bool stationary = true) {
  TrackedObject track;
  track.object_id.uuid[0] = tag;
  track.existence_probability = 0.8F;
  ObjectClassification classification;
  classification.label = ObjectClassification::PEDESTRIAN;
  classification.probability = 1.0F;
  track.classification.push_back(classification);
  track.kinematics.pose_with_covariance.pose.position.x = x;
  track.kinematics.pose_with_covariance.pose.position.y = y;
  track.kinematics.pose_with_covariance.pose.position.z = 0.4;
  track.kinematics.twist_with_covariance.twist.linear.x = vx;
  track.kinematics.twist_with_covariance.twist.linear.y = vy;
  track.kinematics.is_stationary = stationary;
  track.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  track.shape.dimensions.x = length;
  track.shape.dimensions.y = width;
  track.shape.dimensions.z = 1.7;
  return track;
}

TrackedObjects makeTracks(const std::vector<TrackedObject>& objects, int32_t sec = 10, uint32_t nanosec = 699999999) {
  TrackedObjects tracks;
  tracks.header.frame_id = "camera_init";
  tracks.header.stamp.sec = sec;
  tracks.header.stamp.nanosec = nanosec;
  tracks.objects = objects;
  return tracks;
}

// ------------------------------------------------------------------------------------------------ toObstacles

// Opti-Pessi keeps away from a person who is not moving too: they can start walking at any time.
TEST(ToObstacles, StandingPersonIsAnObstacleWithTheSpeedOfItsClass) {
  const TrackedObjects tracks = makeTracks({makeTrack(3.0, 1.0, 7)});
  const DynamicObstacleArray out = toObstacles(tracks, Eigen::Vector2d::Zero(), kShipped);

  EXPECT_EQ(out.header, tracks.header);
  ASSERT_EQ(out.obstacles.size(), 1u);
  const DynamicObstacle& person = out.obstacles.front();
  EXPECT_EQ(person.id.uuid[0], 7);
  EXPECT_EQ(person.label, ObjectClassification::PEDESTRIAN);
  EXPECT_DOUBLE_EQ(person.position.x, 3.0);
  EXPECT_DOUBLE_EQ(person.position.y, 1.0);
  EXPECT_DOUBLE_EQ(person.position.z, 0.4);
  EXPECT_TRUE(person.is_stationary);
  EXPECT_FLOAT_EQ(person.max_speed, 1.5F);
  EXPECT_FLOAT_EQ(person.confidence, 0.8F);
  EXPECT_FLOAT_EQ(person.radius, 0.35F);  // half of 0.5 m + 0.10 m
}

TEST(ToObstacles, MaxSpeedIsTheOneOfTheClassWhateverTheEstimatedVelocity) {
  const DynamicObstacleArray out =
      toObstacles(makeTracks({makeTrack(3.0, 1.0, 1, 0.5, 0.3, 3.0, -0.5, false)}), Eigen::Vector2d::Zero(), kShipped);
  ASSERT_EQ(out.obstacles.size(), 1u);
  EXPECT_FLOAT_EQ(out.obstacles.front().max_speed, 1.5F);
  EXPECT_FALSE(out.obstacles.front().is_stationary);
  EXPECT_DOUBLE_EQ(out.obstacles.front().velocity.x, 3.0);  // for information only
  EXPECT_DOUBLE_EQ(out.obstacles.front().velocity.y, -0.5);
}

TEST(ToObstacles, RadiusIsHalfTheLongerSidePlusTheMarginWithinItsLimits) {
  const DynamicObstacleArray out = toObstacles(
      makeTracks({makeTrack(1.0, 0.0, 1, 0.3, 0.5),    // longer side is the width: 0.25 + 0.10
                  makeTrack(2.0, 0.0, 2, 0.2, 0.2),    // 0.10 + 0.10: not less than radius_min
                  makeTrack(3.0, 0.0, 3, 1.2, 0.4)}),  // 0.60 + 0.10: not more than radius_max
      Eigen::Vector2d::Zero(), kShipped);
  ASSERT_EQ(out.obstacles.size(), 3u);
  EXPECT_FLOAT_EQ(out.obstacles[0].radius, 0.35F);
  EXPECT_FLOAT_EQ(out.obstacles[1].radius, 0.30F);
  EXPECT_FLOAT_EQ(out.obstacles[2].radius, 0.50F);
}

TEST(ToObstacles, ClosestFirstAndNoMoreThanMaxObstacles) {
  // Robot at (10, 0); the tag of each person is its rank by distance
  const DynamicObstacleArray out = toObstacles(
      makeTracks({makeTrack(15.0, 0.0, 4), makeTrack(10.0, 1.0, 1), makeTrack(10.0, -6.0, 5), makeTrack(7.0, 0.0, 3),
                  makeTrack(11.0, 1.5, 2), makeTrack(10.0, 9.0, 6)}),
      Eigen::Vector2d(10.0, 0.0), kShipped);
  ASSERT_EQ(out.obstacles.size(), 4u);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(out.obstacles[i].id.uuid[0], i + 1) << "place " << i;
  }
}

TEST(ToObstacles, PeopleBeyondMaxRangeAreLeftOut) {
  const DynamicObstacleArray out = toObstacles(makeTracks({makeTrack(20.5, 0.0, 1), makeTrack(19.5, 0.0, 2)}),
                                               Eigen::Vector2d(10.0, 0.0), kShipped);
  ASSERT_EQ(out.obstacles.size(), 1u);
  EXPECT_EQ(out.obstacles.front().id.uuid[0], 2);
}

TEST(ToObstacles, NobodyGivesAnEmptyListWithTheHeaderOfTheTracks) {
  const TrackedObjects tracks = makeTracks({});
  const DynamicObstacleArray out = toObstacles(tracks, Eigen::Vector2d::Zero(), kShipped);
  EXPECT_TRUE(out.obstacles.empty());
  EXPECT_EQ(out.header, tracks.header);
}

// ---------------------------------------------------------------------------------------------------- the node

class ObstacleAdapterTest : public test::RosTest {
 protected:
  void SetUp() override {
    RosTest::SetUp();
    adapter_ = std::make_shared<ObstacleAdapter>(test::shippedConfig("obstacle_adapter.yaml"));
    executor_.add_node(adapter_);
    tracks_ = io_->create_publisher<TrackedObjects>("/perception/tracks", 20);
    odometry_ = io_->create_publisher<Odometry>("/Odometry", 20);
    obstacles_ = collect<DynamicObstacleArray>("/perception/dynamic_obstacles");
    markers_ = collect<MarkerArray>("/perception/dynamic_obstacles_markers");
    ASSERT_TRUE(spinUntil([&] {
      return tracks_->get_subscription_count() == 1 && odometry_->get_subscription_count() == 1 &&
             io_->count_publishers("/perception/dynamic_obstacles") == 1 &&
             io_->count_publishers("/perception/dynamic_obstacles_markers") == 1;
    })) << "the adapter does not read /perception/tracks and /Odometry, or does not publish its two topics";
  }

  std::shared_ptr<ObstacleAdapter> adapter_;
  rclcpp::Publisher<TrackedObjects>::SharedPtr tracks_;
  rclcpp::Publisher<Odometry>::SharedPtr odometry_;
  std::shared_ptr<std::vector<DynamicObstacleArray>> obstacles_;
  std::shared_ptr<std::vector<MarkerArray>> markers_;
};

TEST_F(ObstacleAdapterTest, PeopleAroundTheRobotComeOutAtTheStampOfTheirTracks) {
  odometry_->publish(makeOdometry(10, 699999999, Eigen::Vector3d(10.0, 0.0, 0.0)));
  spinFor(100ms);
  tracks_->publish(makeTracks({makeTrack(25.0, 0.0, 2), makeTrack(12.0, 0.0, 1)}));  // 15 m and 2 m from the robot

  ASSERT_TRUE(spinUntil([&] { return !obstacles_->empty(); }));
  const DynamicObstacleArray& out = obstacles_->front();
  EXPECT_EQ(out.header.frame_id, "camera_init");
  EXPECT_EQ(out.header.stamp.sec, 10);
  EXPECT_EQ(out.header.stamp.nanosec, 699999999u);
  ASSERT_EQ(out.obstacles.size(), 1u);
  EXPECT_EQ(out.obstacles.front().id.uuid[0], 1);
}

// The consumer tells a perception that stopped from a scene without people by the messages still coming.
TEST_F(ObstacleAdapterTest, EveryTracksMessageGetsAnAnswerEvenWithoutPeople) {
  odometry_->publish(makeOdometry(10, 0));
  spinFor(100ms);
  for (uint32_t i = 0; i < 5; ++i) {
    tracks_->publish(makeTracks({}, 10, i * 100000000u));
    ASSERT_TRUE(spinUntil([&] { return obstacles_->size() > i; })) << "tracks " << i << " unanswered";
  }
  spinFor(200ms);
  ASSERT_EQ(obstacles_->size(), 5u);
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_TRUE((*obstacles_)[i].obstacles.empty());
    EXPECT_EQ((*obstacles_)[i].header.stamp.nanosec, i * 100000000u);
  }
}

// Which people are in range, and which are the closest, cannot be told without the robot.
TEST_F(ObstacleAdapterTest, WithoutOdometryNothingIsPublished) {
  tracks_->publish(makeTracks({makeTrack(2.0, 0.0)}));
  spinFor(400ms);
  EXPECT_TRUE(obstacles_->empty());
}

TEST_F(ObstacleAdapterTest, MarkersDrawTheRadiusAndTheRoomThePessimisticConstraintTakes) {
  odometry_->publish(makeOdometry(10, 699999999));
  spinFor(100ms);
  tracks_->publish(makeTracks({makeTrack(3.0, 1.0)}));
  ASSERT_TRUE(spinUntil([&] { return !markers_->empty(); }));

  std::vector<double> radii;
  for (const Marker& marker : markers_->front().markers) {
    if (marker.type != Marker::LINE_STRIP) {
      continue;
    }
    EXPECT_EQ(marker.header.frame_id, "camera_init");
    EXPECT_EQ(marker.header.stamp.nanosec, 699999999u);
    ASSERT_GT(marker.points.size(), 8u);
    for (const auto& point : marker.points) {  // a circle around the person
      EXPECT_NEAR(std::hypot(point.x - 3.0, point.y - 1.0), std::hypot(marker.points[0].x - 3.0, marker.points[0].y - 1.0),
                  1e-6);
    }
    radii.push_back(std::hypot(marker.points[0].x - 3.0, marker.points[0].y - 1.0));
  }
  std::sort(radii.begin(), radii.end());
  ASSERT_EQ(radii.size(), 2u);
  EXPECT_NEAR(radii[0], 0.35, 1e-6);               // the obstacle
  EXPECT_NEAR(radii[1], 0.35 + 1.5 * 2.1, 1e-6);   // radius + max_speed * marker_horizon
}

}  // namespace
