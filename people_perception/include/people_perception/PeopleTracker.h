#pragma once

// Tracker of block B5 of docs/perception_pipeline.md, without ROS: people on a plane, one Kalman filter per person
// (position and velocity, constant velocity model), detections matched to tracks by the smallest total distance.
//
// A track starts tentative from a detection no track took, is confirmed once it was detected min_hits times in its
// last hits_window scans with a mean score of min_score_confirm, coasts on its prediction while no detection matches
// it, and is dropped after max_age without one. A person who stands still keeps the track as long as the detector
// sees them.

#include <array>
#include <cstdint>
#include <deque>
#include <random>
#include <vector>

#include <Eigen/Dense>

namespace people_perception {

struct Detection {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();    // box centre in the tracking frame [m]
  Eigen::Vector3d dimensions = Eigen::Vector3d::Zero();  // length, width, height [m]
  double yaw = 0.0;
  double score = 0.0;
};

// All from config/people_tracker.yaml, see there.
struct TrackerParams {
  double gate_distance = 0.0;
  double min_score_new_track = 0.0;
  double min_score_confirm = 0.0;
  int min_hits = 0;
  int hits_window = 0;
  double max_age = 0.0;
  double stationary_speed = 0.0;
  double stationary_hysteresis = 0.0;
  double max_plausible_speed = 0.0;
  double process_noise_acceleration = 0.0;
  double measurement_noise_position = 0.0;
  double initial_speed_std = 0.0;
  double shape_smoothing = 0.0;
  double backward_jump_reset = 0.0;
};

struct Track {
  using Id = std::array<uint8_t, 16>;

  Id id{};
  bool confirmed = false;
  double time_since_update = 0.0;  // [s] since a detection matched it; 0: one of the last scan did
  Eigen::Vector4d state = Eigen::Vector4d::Zero();  // x y vx vy
  Eigen::Matrix4d covariance = Eigen::Matrix4d::Zero();
  double z = 0.0;  // box centre
  Eigen::Vector3d dimensions = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double score = 0.0;  // mean score of the detections that matched it
  bool stationary = false;

  Eigen::Vector2d position() const { return state.head<2>(); }
  Eigen::Vector2d velocity() const { return state.tail<2>(); }

  // Bookkeeping of the tracker
  std::deque<bool> hits;  // one entry per scan of its last hits_window: matched or not
  int detections = 0;
  Eigen::Vector2d last_detection = Eigen::Vector2d::Zero();
  double last_detection_stamp = 0.0;
};

class PeopleTracker {
 public:
  explicit PeopleTracker(const TrackerParams& params);

  /// The detections of the scan measured at `stamp` [s]. False: the scan is not after the last one and was ignored. A
  /// stamp more than backward_jump_reset before the last one (a bag started again) drops every track and is taken.
  bool update(double stamp, const std::vector<Detection>& detections);

  /// Confirmed, tentative and coasting tracks, at the stamp of the last scan taken.
  const std::vector<Track>& tracks() const { return tracks_; }

 private:
  void predict(Track& track, double dt) const;
  void correct(Track& track, const Detection& detection, double stamp) const;
  Track start(const Detection& detection, double stamp);

  TrackerParams params_;
  std::vector<Track> tracks_;
  bool has_stamp_ = false;
  double stamp_ = 0.0;
  std::mt19937_64 id_generator_{std::random_device{}()};
};

}  // namespace people_perception
