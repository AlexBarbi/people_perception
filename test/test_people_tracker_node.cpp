#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <legged_controllers/msg/obstacle_array.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "people_perception/PeopleTrackerNode.h"

#include "RosTest.h"

using autoware_perception_msgs::msg::DetectedObject;
using autoware_perception_msgs::msg::DetectedObjects;
using autoware_perception_msgs::msg::ObjectClassification;
using autoware_perception_msgs::msg::Shape;
using autoware_perception_msgs::msg::TrackedObject;
using autoware_perception_msgs::msg::TrackedObjects;
using legged_controllers::msg::Obstacle;
using legged_controllers::msg::ObstacleArray;
using people_perception::PeopleTrackerNode;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;
using namespace std::chrono_literals;

namespace {

const char* const kObstacles = "/opti_pessi/obstacles";

DetectedObject makeObject(double x, double y, uint8_t label = ObjectClassification::PEDESTRIAN, float score = 0.8F) {
  DetectedObject object;
  object.existence_probability = score;
  ObjectClassification classification;
  classification.label = label;
  classification.probability = 1.0F;
  object.classification.push_back(classification);
  object.kinematics.pose_with_covariance.pose.position.x = x;
  object.kinematics.pose_with_covariance.pose.position.y = y;
  object.kinematics.pose_with_covariance.pose.position.z = 0.4;
  object.kinematics.pose_with_covariance.pose.orientation.w = 1.0;
  object.shape.type = Shape::BOUNDING_BOX;
  object.shape.dimensions.x = 0.5;
  object.shape.dimensions.y = 0.4;
  object.shape.dimensions.z = 1.7;
  return object;
}

// The scan number `scan` of a 10 Hz detector, in the frame of the detector cloud.
DetectedObjects makeDetections(int scan, const std::vector<DetectedObject>& objects,
                               const std::string& frame = "sensor_level") {
  DetectedObjects msg;
  msg.header.frame_id = frame;
  msg.header.stamp.sec = 100 + scan / 10;
  msg.header.stamp.nanosec = static_cast<uint32_t>(scan % 10) * 100000000u;
  msg.objects = objects;
  return msg;
}

geometry_msgs::msg::TransformStamped transform(const std::string& parent, const std::string& child, double x, double y,
                                               double z, double yaw) {
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = parent;
  transform.child_frame_id = child;
  transform.transform.translation.x = x;
  transform.transform.translation.y = y;
  transform.transform.translation.z = z;
  transform.transform.rotation.z = std::sin(yaw / 2.0);
  transform.transform.rotation.w = std::cos(yaw / 2.0);
  return transform;
}

class PeopleTrackerNodeTest : public test::RosTest {
 protected:
  void SetUp() override {
    RosTest::SetUp();
    tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(io_);
    // sensor_level 10 m and 5 m from the origin of camera_init, 0.47 m up, turned 90 deg to the left; camera_init at
    // (1, 2, 0.5) in odom, turned 90 deg to the left too
    tf_->sendTransform({transform("camera_init", "sensor_level", 10.0, 5.0, 0.47, M_PI / 2.0),
                        transform("odom", "camera_init", 1.0, 2.0, 0.5, M_PI / 2.0)});

    tracker_ = std::make_shared<PeopleTrackerNode>(options_);
    executor_.add_node(tracker_);
    detections_ = io_->create_publisher<DetectedObjects>("/perception/detector/objects", 20);
    tracks_ = collect<TrackedObjects>("/perception/tracks");
    markers_ = collect<MarkerArray>("/perception/tracks_markers");
    obstacles_ = collect<ObstacleArray>(kObstacles);
    ASSERT_TRUE(spinUntil([&] {
      return detections_->get_subscription_count() == 1 && io_->count_publishers("/perception/tracks") == 1 &&
             io_->count_publishers("/perception/tracks_markers") == 1 &&
             io_->count_publishers(kObstacles) == (obstacles_enabled_ ? 1u : 0u);
    })) << "the tracker does not read /perception/detector/objects or does not publish its topics";

    // The tracker hears /tf_static a moment after it starts and drops the detections it cannot place: wait until it
    // answers. The probes are long before the scans of the tests and hold nobody.
    for (uint32_t i = 0; i < 100 && tracks_->empty(); ++i) {
      DetectedObjects probe;
      probe.header.frame_id = "sensor_level";
      probe.header.stamp.sec = 1;
      probe.header.stamp.nanosec = i * 1000000u;
      detections_->publish(probe);
      spinFor(100ms);
    }
    ASSERT_FALSE(tracks_->empty()) << "the tracker never answered: no TF camera_init <- sensor_level?";
    spinFor(100ms);
    tracks_->clear();
    markers_->clear();
    obstacles_->clear();
  }

  // One detector message, and the answer of the tracker to it.
  void scan(const DetectedObjects& msg) {
    const size_t before = tracks_->size();
    detections_->publish(msg);
    ASSERT_TRUE(spinUntil([&] {
      return tracks_->size() > before && markers_->size() > before &&
             (!obstacles_come_ || obstacles_->size() > before);
    })) << "no answer to the detections of " << msg.header.stamp.sec << "." << msg.header.stamp.nanosec;
  }

  rclcpp::NodeOptions options_ = test::shippedConfig("people_tracker.yaml");
  bool obstacles_enabled_ = true;  // the tracker publishes on kObstacles
  bool obstacles_come_ = true;     // and answers every scan there
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_;
  std::shared_ptr<PeopleTrackerNode> tracker_;
  rclcpp::Publisher<DetectedObjects>::SharedPtr detections_;
  std::shared_ptr<std::vector<TrackedObjects>> tracks_;
  std::shared_ptr<std::vector<MarkerArray>> markers_;
  std::shared_ptr<std::vector<ObstacleArray>> obstacles_;
};

TEST_F(PeopleTrackerNodeTest, TracksAreInTheOdometryFrameAtTheStampOfTheDetections) {
  for (int i = 0; i < 3; ++i) {
    scan(makeDetections(i, {makeObject(2.0, 0.0)}));
  }
  ASSERT_EQ(tracks_->size(), 3u);
  for (int i = 0; i < 3; ++i) {
    const TrackedObjects& tracks = (*tracks_)[static_cast<size_t>(i)];
    EXPECT_EQ(tracks.header.frame_id, "camera_init");
    EXPECT_EQ(tracks.header.stamp, makeDetections(i, {}).header.stamp);
    ASSERT_EQ(tracks.objects.size(), 1u);  // tentative ones too: the filter takes them out
    // 2 m ahead of a sensor at (10, 5, 0.47) that looks along +y
    const auto& position = tracks.objects.front().kinematics.pose_with_covariance.pose.position;
    EXPECT_NEAR(position.x, 10.0, 1e-6);
    EXPECT_NEAR(position.y, 7.0, 1e-6);
    EXPECT_NEAR(position.z, 0.87, 1e-6);
  }
}

TEST_F(PeopleTrackerNodeTest, EveryDetectionMessageGetsAnAnswerEvenWithoutPeople) {
  for (int i = 0; i < 3; ++i) {
    scan(makeDetections(i, {}));
  }
  spinFor(200ms);
  ASSERT_EQ(tracks_->size(), 3u);
  ASSERT_EQ(obstacles_->size(), 3u);  // an empty array clears the obstacles of the controller
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE((*tracks_)[i].objects.empty());
    EXPECT_EQ((*tracks_)[i].header.stamp, makeDetections(static_cast<int>(i), {}).header.stamp);
    EXPECT_TRUE((*obstacles_)[i].obstacles.empty());
  }
}

TEST_F(PeopleTrackerNodeTest, TrackedObjectDescribesAWalkingPerson) {
  // 1 m/s along the x of the sensor, which is the y of camera_init
  for (int i = 0; i <= 20; ++i) {
    scan(makeDetections(i, {makeObject(2.0 + 0.1 * i, 0.0)}));
  }
  ASSERT_EQ(tracks_->back().objects.size(), 1u);
  const TrackedObject& person = tracks_->back().objects.front();

  ASSERT_EQ(person.classification.size(), 1u);
  EXPECT_EQ(person.classification.front().label, ObjectClassification::PEDESTRIAN);
  EXPECT_NEAR(person.existence_probability, 0.8, 1e-6);
  EXPECT_EQ(person.shape.type, Shape::BOUNDING_BOX);
  EXPECT_NEAR(person.shape.dimensions.x, 0.5, 1e-6);
  EXPECT_NEAR(person.shape.dimensions.y, 0.4, 1e-6);
  EXPECT_NEAR(person.shape.dimensions.z, 1.7, 1e-6);
  EXPECT_NEAR(person.kinematics.twist_with_covariance.twist.linear.x, 0.0, 0.05);
  EXPECT_NEAR(person.kinematics.twist_with_covariance.twist.linear.y, 1.0, 0.05);
  // Position and velocity uncertainty of the filter, for the cylinders of dynamic_points_filter
  EXPECT_GT(person.kinematics.pose_with_covariance.covariance[0], 0.0);
  EXPECT_GT(person.kinematics.pose_with_covariance.covariance[7], 0.0);
  EXPECT_GT(person.kinematics.twist_with_covariance.covariance[0], 0.0);
  EXPECT_GT(person.kinematics.twist_with_covariance.covariance[7], 0.0);
}

TEST_F(PeopleTrackerNodeTest, DetectionsWithoutATransformAreDropped) {
  detections_->publish(makeDetections(0, {makeObject(2.0, 0.0)}, "a_frame_nobody_publishes"));
  spinFor(400ms);
  EXPECT_TRUE(tracks_->empty());
  EXPECT_TRUE(obstacles_->empty());
}

TEST_F(PeopleTrackerNodeTest, ObjectsOfOtherClassesAreNotPeople) {
  for (int i = 0; i < 5; ++i) {
    scan(makeDetections(i, {makeObject(2.0, 0.0, ObjectClassification::CAR)}));
  }
  EXPECT_TRUE(tracks_->back().objects.empty());
  EXPECT_TRUE(obstacles_->back().obstacles.empty());
}

TEST_F(PeopleTrackerNodeTest, MarkersShowEveryTrackInTheOdometryFrame) {
  for (int i = 0; i < 3; ++i) {
    scan(makeDetections(i, {makeObject(2.0, 0.0), makeObject(4.0, 3.0)}));
  }
  const auto& markers = markers_->back().markers;
  const auto boxes = std::count_if(markers.begin(), markers.end(), [](const Marker& m) { return m.type == Marker::CUBE; });
  EXPECT_EQ(boxes, 2);
  for (const Marker& marker : markers) {
    if (marker.action == Marker::DELETEALL) {
      continue;
    }
    EXPECT_EQ(marker.header.frame_id, "camera_init");
    EXPECT_EQ(marker.header.stamp, makeDetections(2, {}).header.stamp);
  }
  const auto box = std::find_if(markers.begin(), markers.end(), [](const Marker& m) { return m.type == Marker::CUBE; });
  ASSERT_NE(box, markers.end());
  EXPECT_NEAR(box->scale.x, 0.5, 1e-6);
  EXPECT_NEAR(box->scale.z, 1.7, 1e-6);
}

// ----------------------------------------------------------------------------------- the obstacles of the controller

TEST_F(PeopleTrackerNodeTest, ConfirmedPeopleAreHumansInOdomAtTheStampOfTheirScan) {
  for (int i = 0; i < 3; ++i) {
    scan(makeDetections(i, {makeObject(2.0, 0.0)}));
  }
  ASSERT_EQ(obstacles_->size(), 3u);
  // Confirmed on the third detection: only then an obstacle
  EXPECT_TRUE((*obstacles_)[0].obstacles.empty());
  EXPECT_TRUE((*obstacles_)[1].obstacles.empty());
  const ObstacleArray& out = (*obstacles_)[2];
  EXPECT_EQ(out.header.frame_id, "odom");  // the controller ignores any other frame
  EXPECT_EQ(out.header.stamp, makeDetections(2, {}).header.stamp);
  ASSERT_EQ(out.obstacles.size(), 1u);
  EXPECT_EQ(out.obstacles.front().type, Obstacle::HUMAN);
  // (10, 7, 0.87) in camera_init, which is turned a quarter left in odom and sits at (1, 2, 0.5)
  EXPECT_NEAR(out.obstacles.front().position.x, -6.0, 1e-6);
  EXPECT_NEAR(out.obstacles.front().position.y, 12.0, 1e-6);
  EXPECT_NEAR(out.obstacles.front().position.z, 1.37, 1e-6);
}

// Each obstacle takes a slot of the controller, and its solve time grows with them.
TEST_F(PeopleTrackerNodeTest, PeopleBeyondMaxRangeOfTheSensorAreNoObstacles) {
  for (int i = 0; i < 3; ++i) {
    scan(makeDetections(i, {makeObject(9.5, 0.0), makeObject(0.0, -10.5)}));
  }
  ASSERT_EQ(tracks_->back().objects.size(), 2u);
  ASSERT_EQ(obstacles_->back().obstacles.size(), 1u);
  // 9.5 m ahead of the sensor: (10, 14.5) in camera_init, (-13.5, 12) in odom
  EXPECT_NEAR(obstacles_->back().obstacles.front().position.x, -13.5, 1e-6);
  EXPECT_NEAR(obstacles_->back().obstacles.front().position.y, 12.0, 1e-6);
}

// Opti-Pessi has to know a person is there at zero speed too.
TEST_F(PeopleTrackerNodeTest, StandingPersonStaysAmongTheObstacles) {
  for (int i = 0; i <= 50; ++i) {
    scan(makeDetections(i, {makeObject(3.0, -1.0)}));
  }
  EXPECT_EQ(obstacles_->back().obstacles.size(), 1u);
}

// The scenario on the tuning that ships: walk, stand 10 s, walk again, one track. A track that replaced it would be
// tentative for its first scans, and missing from the obstacles.
TEST_F(PeopleTrackerNodeTest, WalkStopWalkStaysOneConfirmedTrackWithTheShippedTuning) {
  const auto walked = [](double t) { return t < 3.0 ? t : (t < 13.0 ? 3.0 : t - 10.0); };
  for (int i = 0; i <= 160; ++i) {
    scan(makeDetections(i, {makeObject(2.0 + 1.2 * walked(0.1 * i), 0.5)}));
  }
  for (size_t i = 0; i < tracks_->size(); ++i) {
    ASSERT_EQ((*tracks_)[i].objects.size(), 1u) << "scan " << i;
    if (i >= 2) {  // confirmed on the third detection
      ASSERT_EQ((*obstacles_)[i].obstacles.size(), 1u) << "the track was lost and started again before scan " << i;
    }
  }
}

// Until FAST-LIO aligns its map to the legged estimator there is no odom: nothing goes to the controller, not an empty
// array that would tell it there is nobody.
class PeopleTrackerNodeWithoutOdomTest : public PeopleTrackerNodeTest {
 protected:
  PeopleTrackerNodeWithoutOdomTest() {
    options_.append_parameter_override("obstacles.frame", "a_frame_nobody_publishes");
    obstacles_come_ = false;
  }
};

TEST_F(PeopleTrackerNodeWithoutOdomTest, NoObstaclesWithoutTheTransformToTheFrameOfTheController) {
  ASSERT_EQ(io_->count_publishers(kObstacles), 1u);
  for (int i = 0; i < 5; ++i) {
    scan(makeDetections(i, {makeObject(2.0, 0.0)}));
  }
  spinFor(200ms);
  EXPECT_EQ(tracks_->back().objects.size(), 1u);
  EXPECT_TRUE(obstacles_->empty());
}

// opti_pessi_obstacles:=false leaves the topic to another publisher (yolo26_rgbd_detect).
class PeopleTrackerNodeWithoutObstaclesTest : public PeopleTrackerNodeTest {
 protected:
  PeopleTrackerNodeWithoutObstaclesTest() {
    options_.append_parameter_override("obstacles.enabled", false);
    obstacles_enabled_ = obstacles_come_ = false;
  }
};

TEST_F(PeopleTrackerNodeWithoutObstaclesTest, ObstaclesDisabledLeavesTheTopicToOthers) {
  for (int i = 0; i < 3; ++i) {
    scan(makeDetections(i, {makeObject(2.0, 0.0)}));
  }
  EXPECT_EQ(io_->count_publishers(kObstacles), 0u);
  EXPECT_TRUE(obstacles_->empty());
}

}  // namespace
