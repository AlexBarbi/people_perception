#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "people_perception/PeopleTracker.h"

using people_perception::Detection;
using people_perception::PeopleTracker;
using people_perception::Track;
using people_perception::TrackerParams;

namespace {

const double kScan = 0.1;  // the Airy scans at 10 Hz

// The values of config/people_tracker.yaml.
TrackerParams shippedParams() {
  TrackerParams p;
  p.gate_distance = 1.0;
  p.min_score_new_track = 0.25;
  p.min_score_confirm = 0.40;
  p.min_hits = 3;
  p.hits_window = 5;
  p.max_age = 1.0;
  p.stationary_speed = 0.2;
  p.stationary_hysteresis = 0.1;
  p.max_plausible_speed = 5.0;
  p.process_noise_acceleration = 1.5;
  p.measurement_noise_position = 0.1;
  p.initial_speed_std = 2.0;
  p.shape_smoothing = 0.2;
  p.backward_jump_reset = 2.0;
  return p;
}

Detection person(double x, double y, double score = 0.8, double height = 1.7) {
  Detection d;
  d.position = Eigen::Vector3d(x, y, 0.4);
  d.dimensions = Eigen::Vector3d(0.5, 0.5, height);
  d.score = score;
  return d;
}

std::vector<const Track*> confirmed(const PeopleTracker& tracker) {
  std::vector<const Track*> out;
  for (const Track& track : tracker.tracks()) {
    if (track.confirmed) {
      out.push_back(&track);
    }
  }
  return out;
}

// Scans first .. last of one person at position(t), t = scan * kScan.
template <typename Position>
void see(PeopleTracker& tracker, int first, int last, Position position) {
  for (int scan = first; scan <= last; ++scan) {
    const Eigen::Vector2d p = position(scan * kScan);
    ASSERT_TRUE(tracker.update(scan * kScan, {person(p.x(), p.y())}));
  }
}

void seeNobody(PeopleTracker& tracker, int first, int last) {
  for (int scan = first; scan <= last; ++scan) {
    ASSERT_TRUE(tracker.update(scan * kScan, {}));
  }
}

TEST(PeopleTracker, SingleIsolatedDetectionNeverBecomesAConfirmedTrack) {
  PeopleTracker tracker(shippedParams());
  ASSERT_TRUE(tracker.update(0.0, {person(3.0, 1.0)}));
  ASSERT_EQ(tracker.tracks().size(), 1u);  // tentative: the filter already takes its points out
  EXPECT_FALSE(tracker.tracks().front().confirmed);
  for (int scan = 1; scan <= 20; ++scan) {
    ASSERT_TRUE(tracker.update(scan * kScan, {}));
    EXPECT_TRUE(confirmed(tracker).empty()) << "scan " << scan;
  }
  EXPECT_TRUE(tracker.tracks().empty());
}

TEST(PeopleTracker, PersonSeenOnThreeScansIsConfirmedOnTheThird) {
  PeopleTracker tracker(shippedParams());
  tracker.update(0.0, {person(3.0, 1.0)});
  tracker.update(0.1, {person(3.0, 1.0)});
  EXPECT_TRUE(confirmed(tracker).empty());
  tracker.update(0.2, {person(3.0, 1.0)});
  EXPECT_EQ(confirmed(tracker).size(), 1u);
}

TEST(PeopleTracker, DetectionBelowTheNewTrackScoreStartsNothing) {
  PeopleTracker tracker(shippedParams());
  tracker.update(0.0, {person(3.0, 1.0, 0.2)});
  EXPECT_TRUE(tracker.tracks().empty());
}

TEST(PeopleTracker, LowMeanScoreStaysTentative) {
  PeopleTracker tracker(shippedParams());
  for (int scan = 0; scan < 10; ++scan) {
    tracker.update(scan * kScan, {person(3.0, 1.0, 0.3)});
  }
  ASSERT_EQ(tracker.tracks().size(), 1u);
  EXPECT_FALSE(tracker.tracks().front().confirmed);
}

// The tracks carry no identity. A track that lived on is told from one that replaced it by its count of detections:
// it only grows, and a new track starts at 1.
TEST(PeopleTracker, TrackSurvivesAHalfSecondGap) {
  PeopleTracker tracker(shippedParams());
  const auto walk = [](double t) { return Eigen::Vector2d(1.2 * t, 2.0); };
  see(tracker, 0, 15, walk);
  ASSERT_EQ(confirmed(tracker).size(), 1u);
  EXPECT_EQ(confirmed(tracker).front()->detections, 16);

  seeNobody(tracker, 16, 20);  // 0.5 s without a detection
  ASSERT_EQ(tracker.tracks().size(), 1u);
  EXPECT_EQ(tracker.tracks().front().detections, 16);
  EXPECT_NEAR(tracker.tracks().front().time_since_update, 0.5, 1e-6);

  see(tracker, 21, 30, walk);
  ASSERT_EQ(tracker.tracks().size(), 1u) << "the person came back as another track";
  EXPECT_EQ(tracker.tracks().front().detections, 26) << "the person came back as another track";
  EXPECT_TRUE(tracker.tracks().front().confirmed);
  EXPECT_DOUBLE_EQ(tracker.tracks().front().time_since_update, 0.0);
}

TEST(PeopleTracker, VelocityIsWithin20CentimetersPerSecondAfterOneSecond) {
  PeopleTracker tracker(shippedParams());
  std::mt19937 random(7);
  std::normal_distribution<double> noise(0.0, 0.03);
  const Eigen::Vector2d velocity(1.0, 0.5);
  for (int scan = 0; scan <= 10; ++scan) {
    const Eigen::Vector2d p = Eigen::Vector2d(4.0, -1.0) + velocity * (scan * kScan);
    tracker.update(scan * kScan, {person(p.x() + noise(random), p.y() + noise(random))});
  }
  ASSERT_EQ(tracker.tracks().size(), 1u);
  EXPECT_LT((tracker.tracks().front().velocity() - velocity).norm(), 0.2);
}

TEST(PeopleTracker, TrackIsDroppedAfterMaxAgeWithoutDetections) {
  PeopleTracker tracker(shippedParams());
  see(tracker, 0, 10, [](double) { return Eigen::Vector2d(3.0, 1.0); });
  seeNobody(tracker, 11, 20);  // 1.0 s: still coasting
  EXPECT_EQ(tracker.tracks().size(), 1u);
  seeNobody(tracker, 21, 21);
  EXPECT_TRUE(tracker.tracks().empty());
}

TEST(PeopleTracker, StandingPersonStaysConfirmedAndStationary) {
  PeopleTracker tracker(shippedParams());
  std::mt19937 random(3);
  std::normal_distribution<double> noise(0.0, 0.03);
  for (int scan = 0; scan <= 100; ++scan) {
    tracker.update(scan * kScan, {person(2.0 + noise(random), -1.5 + noise(random))});
    if (scan >= 10) {
      ASSERT_EQ(tracker.tracks().size(), 1u) << "scan " << scan;
      EXPECT_EQ(tracker.tracks().front().detections, scan + 1) << "another track at scan " << scan;
      EXPECT_TRUE(tracker.tracks().front().confirmed) << "scan " << scan;
      EXPECT_TRUE(tracker.tracks().front().stationary) << "scan " << scan;
    }
  }
}

TEST(PeopleTracker, WalkStopWalkKeepsTheTrack) {
  PeopleTracker tracker(shippedParams());
  // 3 s at 1.2 m/s, 10 s standing, 3 s at 1.2 m/s
  const auto path = [](double t) {
    const double walked = t < 3.0 ? t : (t < 13.0 ? 3.0 : t - 10.0);
    return Eigen::Vector2d(1.2 * walked, 0.5);
  };
  see(tracker, 0, 30, path);
  ASSERT_EQ(confirmed(tracker).size(), 1u);
  EXPECT_FALSE(confirmed(tracker).front()->stationary);

  see(tracker, 31, 130, path);
  ASSERT_EQ(tracker.tracks().size(), 1u);
  EXPECT_EQ(tracker.tracks().front().detections, 131);
  EXPECT_TRUE(tracker.tracks().front().stationary);
  EXPECT_TRUE(tracker.tracks().front().confirmed);

  see(tracker, 131, 160, path);
  ASSERT_EQ(tracker.tracks().size(), 1u);
  EXPECT_EQ(tracker.tracks().front().detections, 161);
  EXPECT_FALSE(tracker.tracks().front().stationary);
}

TEST(PeopleTracker, SlowDriftOfAStandingPersonStaysStationary) {
  PeopleTracker tracker(shippedParams());
  // Standing, then 0.25 m/s: above stationary_speed, below stationary_speed + stationary_hysteresis
  const auto path = [](double t) { return Eigen::Vector2d(t < 3.0 ? 0.0 : 0.25 * (t - 3.0), 0.0); };
  see(tracker, 0, 30, path);
  ASSERT_TRUE(tracker.tracks().front().stationary);
  for (int scan = 31; scan <= 90; ++scan) {
    see(tracker, scan, scan, path);
    EXPECT_TRUE(tracker.tracks().front().stationary) << "scan " << scan;
  }
}

TEST(PeopleTracker, WalkerSlowingToAShuffleIsStillMoving) {
  PeopleTracker tracker(shippedParams());
  // 1.2 m/s, then down to 0.25 m/s over 3 s, then 0.25 m/s: never below stationary_speed
  const auto speed = [](double t) { return t < 2.0 ? 1.2 : (t < 5.0 ? 1.2 - 0.95 * (t - 2.0) / 3.0 : 0.25); };
  double x = 0.0;
  for (int scan = 0; scan <= 90; ++scan) {
    tracker.update(scan * kScan, {person(x, 0.0)});
    x += speed(scan * kScan) * kScan;
    if (scan >= 10) {
      ASSERT_EQ(tracker.tracks().size(), 1u);
      EXPECT_FALSE(tracker.tracks().front().stationary) << "scan " << scan;
    }
  }
}

TEST(PeopleTracker, TwoPeoplePassingEachOtherKeepTheirTracks) {
  PeopleTracker tracker(shippedParams());
  // Opposite ways on lanes 0.8 m apart: at the middle each is within the gate of the other's track
  const auto left = [](double t) { return Eigen::Vector2d(-3.0 + 1.2 * t, 0.4); };
  const auto right = [](double t) { return Eigen::Vector2d(3.0 - 1.2 * t, -0.4); };
  size_t left_track = 0;  // its place in tracks(), which keep the order they started in
  for (int scan = 0; scan <= 50; ++scan) {
    const double t = scan * kScan;
    const Detection a = person(left(t).x(), left(t).y());
    const Detection b = person(right(t).x(), right(t).y());
    tracker.update(t, scan % 2 ? std::vector<Detection>{a, b} : std::vector<Detection>{b, a});
    if (scan == 5) {
      ASSERT_EQ(tracker.tracks().size(), 2u);
      ASSERT_EQ(confirmed(tracker).size(), 2u);
      left_track = tracker.tracks()[0].position().y() > 0 ? 0 : 1;
    }
  }
  ASSERT_EQ(tracker.tracks().size(), 2u);
  for (size_t i = 0; i < 2; ++i) {
    const Track& track = tracker.tracks()[i];
    EXPECT_TRUE(track.confirmed);
    EXPECT_EQ(track.detections, 51) << "one of the two was lost and started again";
    EXPECT_EQ(i == left_track, track.position().y() > 0) << "the two tracks swapped people";
    EXPECT_EQ(i == left_track, track.velocity().x() > 0);
  }
}

// The actors of the Gazebo warehouse (measured on the lidar, 2026-10-05): Gazebo runs a spline through the waypoints of
// a scripted actor, and at the one-second turn at each end of its lane the actor does not stop: it leaves the waypoint
// at 4.5 m/s, swings 1.1 m past it and comes back. With max_plausible_speed 4.0 every turn started a new track.
TEST(PeopleTracker, GazeboActorTurningAtTheEndOfItsLaneKeepsItsTrack) {
  PeopleTracker tracker(shippedParams());
  const auto ease = [](double u) { return -u * u * u + 1.5 * u * u + 0.5 * u; };
  const auto actor = [&](double t) {
    if (t < 7.5) {
      return Eigen::Vector2d(3.5, 4.5 - 9.0 * ease(t / 7.5));  // 9 m lane, 0.6 m/s at its ends, 1.5 m/s in the middle
    }
    if (t < 8.5) {
      const double u = t - 7.5;
      return Eigen::Vector2d(3.5, -4.5 - 4.5 * (u - u * u));  // the swing past the waypoint
    }
    return Eigen::Vector2d(3.5, -4.5 + 9.0 * ease((t - 8.5) / 7.5));
  };
  see(tracker, 0, 10, actor);
  ASSERT_EQ(confirmed(tracker).size(), 1u);
  for (int scan = 11; scan <= 160; ++scan) {
    see(tracker, scan, scan, actor);
    ASSERT_EQ(tracker.tracks().size(), 1u) << "another track at " << scan * kScan << " s";
    EXPECT_EQ(tracker.tracks().front().detections, scan + 1) << "another track at " << scan * kScan << " s";
    EXPECT_DOUBLE_EQ(tracker.tracks().front().time_since_update, 0.0);
  }
}

TEST(PeopleTracker, DetectionThatWouldNeedAnImplausibleSpeedIsAnotherTrack) {
  PeopleTracker tracker(shippedParams());
  see(tracker, 0, 5, [](double) { return Eigen::Vector2d(0.0, 0.0); });
  ASSERT_EQ(confirmed(tracker).size(), 1u);

  tracker.update(0.6, {person(0.9, 0.0)});  // 0.9 m in 0.1 s: 9 m/s, inside the 1 m gate
  ASSERT_EQ(tracker.tracks().size(), 2u);
  const Track& old_track = tracker.tracks()[0];  // tracks() keep the order they started in
  EXPECT_TRUE(old_track.confirmed);
  EXPECT_EQ(old_track.detections, 6);
  EXPECT_NEAR(old_track.position().x(), 0.0, 0.05);
  EXPECT_NEAR(old_track.time_since_update, 0.1, 1e-6);
  const Track& new_track = tracker.tracks()[1];
  EXPECT_FALSE(new_track.confirmed);
  EXPECT_EQ(new_track.detections, 1);
  EXPECT_NEAR(new_track.position().x(), 0.9, 1e-6);
}

TEST(PeopleTracker, DetectionBeyondTheGateIsAnotherTrack) {
  PeopleTracker tracker(shippedParams());
  see(tracker, 0, 5, [](double) { return Eigen::Vector2d(0.0, 0.0); });
  seeNobody(tracker, 6, 9);
  tracker.update(1.0, {person(1.5, 0.0)});  // 1.5 m in 0.5 s is a plausible 3 m/s, but outside the 1 m gate
  EXPECT_EQ(tracker.tracks().size(), 2u);
}

TEST(PeopleTracker, ScanNotAfterTheLastOneIsIgnored) {
  PeopleTracker tracker(shippedParams());
  see(tracker, 0, 10, [](double) { return Eigen::Vector2d(0.0, 0.0); });
  const Track before = tracker.tracks().front();

  EXPECT_FALSE(tracker.update(0.95, {person(5.0, 5.0)}));
  EXPECT_FALSE(tracker.update(1.0, {person(5.0, 5.0)}));
  ASSERT_EQ(tracker.tracks().size(), 1u);
  EXPECT_EQ(tracker.tracks().front().detections, before.detections);
  EXPECT_EQ(tracker.tracks().front().position(), before.position());
}

TEST(PeopleTracker, ClockGoingBackDropsEveryTrack) {
  PeopleTracker tracker(shippedParams());
  for (int scan = 0; scan <= 10; ++scan) {
    tracker.update(100.0 + scan * kScan, {person(0.0, 0.0)});
  }
  ASSERT_EQ(tracker.tracks().front().detections, 11);

  EXPECT_TRUE(tracker.update(1.0, {person(5.0, 5.0)}));  // the bag started again
  ASSERT_EQ(tracker.tracks().size(), 1u);
  EXPECT_EQ(tracker.tracks().front().detections, 1);
  EXPECT_FALSE(tracker.tracks().front().confirmed);
  EXPECT_NEAR(tracker.tracks().front().position().x(), 5.0, 1e-6);
}

// The filter widens the cylinder of a track by its position uncertainty: it has to grow while the track coasts.
TEST(PeopleTracker, PositionUncertaintyGrowsWhileCoasting) {
  PeopleTracker tracker(shippedParams());
  see(tracker, 0, 20, [](double t) { return Eigen::Vector2d(1.2 * t, 0.0); });
  double last = tracker.tracks().front().covariance(0, 0);
  EXPECT_LT(std::sqrt(last), 0.1);  // tracked: within the 0.1 m of a detection
  for (int scan = 21; scan <= 30; ++scan) {
    seeNobody(tracker, scan, scan);
    ASSERT_EQ(tracker.tracks().size(), 1u);
    EXPECT_GT(tracker.tracks().front().covariance(0, 0), last) << "scan " << scan;
    last = tracker.tracks().front().covariance(0, 0);
  }
  // After 1 s the person can be a few steps from the prediction, not across the hall
  EXPECT_GT(std::sqrt(last), 0.3);
  EXPECT_LT(std::sqrt(last), 1.5);
}

TEST(PeopleTracker, BoxIsAveragedOverTheScans) {
  PeopleTracker tracker(shippedParams());
  for (int scan = 0; scan < 60; ++scan) {
    tracker.update(scan * kScan, {person(1.0, 1.0, 0.8, scan % 2 ? 1.6 : 1.8)});
  }
  ASSERT_EQ(tracker.tracks().size(), 1u);
  EXPECT_NEAR(tracker.tracks().front().dimensions.z(), 1.7, 0.05);
  EXPECT_NEAR(tracker.tracks().front().z, 0.4, 1e-6);
}

}  // namespace
