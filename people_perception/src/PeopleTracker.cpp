#include "people_perception/PeopleTracker.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace people_perception {

namespace {

// Cost of a pair that must not be matched: more than any sum of real distances.
constexpr double kForbidden = 1e6;

// Minimum cost assignment (Hungarian algorithm with potentials) of every row to a column of its own, rows <= columns.
// Returns the column of each row.
std::vector<int> assign(const Eigen::MatrixXd& cost) {
  const int rows = static_cast<int>(cost.rows());
  const int columns = static_cast<int>(cost.cols());
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> u(rows + 1, 0.0), v(columns + 1, 0.0);
  std::vector<int> row_of(columns + 1, 0), way(columns + 1, 0);
  for (int row = 1; row <= rows; ++row) {
    row_of[0] = row;
    int column = 0;
    std::vector<double> slack(columns + 1, infinity);
    std::vector<bool> used(columns + 1, false);
    do {
      used[column] = true;
      const int current = row_of[column];
      double delta = infinity;
      int next = 0;
      for (int j = 1; j <= columns; ++j) {
        if (used[j]) {
          continue;
        }
        const double reduced = cost(current - 1, j - 1) - u[current] - v[j];
        if (reduced < slack[j]) {
          slack[j] = reduced;
          way[j] = column;
        }
        if (slack[j] < delta) {
          delta = slack[j];
          next = j;
        }
      }
      for (int j = 0; j <= columns; ++j) {
        if (used[j]) {
          u[row_of[j]] += delta;
          v[j] -= delta;
        } else {
          slack[j] -= delta;
        }
      }
      column = next;
    } while (row_of[column] != 0);
    do {
      const int previous = way[column];
      row_of[column] = row_of[previous];
      column = previous;
    } while (column != 0);
  }
  std::vector<int> column_of(rows, -1);
  for (int j = 1; j <= columns; ++j) {
    if (row_of[j] != 0) {
      column_of[row_of[j] - 1] = j - 1;
    }
  }
  return column_of;
}

double wrapAngle(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }

}  // namespace

PeopleTracker::PeopleTracker(const TrackerParams& params) : params_(params) {}

bool PeopleTracker::update(double stamp, const std::vector<Detection>& detections) {
  if (has_stamp_ && stamp <= stamp_) {
    if (stamp_ - stamp <= params_.backward_jump_reset) {
      return false;
    }
    // The clock went back (a bag started again): these tracks belong to another time
    tracks_.clear();
    has_stamp_ = false;
  }
  const double dt = has_stamp_ ? stamp - stamp_ : 0.0;
  stamp_ = stamp;
  has_stamp_ = true;

  for (Track& track : tracks_) {
    predict(track, dt);
    track.time_since_update = stamp - track.last_detection_stamp;
  }

  // Tracks and detections paired for the smallest total distance on the plane. A pair is out if the detection is
  // farther than the gate from where the track is expected, or if getting there from the last detection of the track
  // would have taken more than max_plausible_speed.
  std::vector<int> detection_of(tracks_.size(), -1);
  std::vector<bool> taken(detections.size(), false);
  if (!tracks_.empty() && !detections.empty()) {
    Eigen::MatrixXd cost(tracks_.size(), detections.size());
    for (size_t t = 0; t < tracks_.size(); ++t) {
      for (size_t d = 0; d < detections.size(); ++d) {
        const Eigen::Vector2d detection = detections[d].position.head<2>();
        const double distance = (detection - tracks_[t].position()).norm();
        const double speed = (detection - tracks_[t].last_detection).norm() / tracks_[t].time_since_update;
        cost(t, d) = distance <= params_.gate_distance && speed <= params_.max_plausible_speed ? distance : kForbidden;
      }
    }
    const bool more_tracks = tracks_.size() > detections.size();
    const std::vector<int> column_of = assign(more_tracks ? Eigen::MatrixXd(cost.transpose()) : cost);
    for (size_t row = 0; row < column_of.size(); ++row) {
      const size_t t = more_tracks ? static_cast<size_t>(column_of[row]) : row;
      const size_t d = more_tracks ? row : static_cast<size_t>(column_of[row]);
      if (cost(t, d) < kForbidden) {
        detection_of[t] = static_cast<int>(d);
        taken[d] = true;
      }
    }
  }

  for (size_t t = 0; t < tracks_.size(); ++t) {
    Track& track = tracks_[t];
    const bool matched = detection_of[t] >= 0;
    if (matched) {
      correct(track, detections[static_cast<size_t>(detection_of[t])], stamp);
    }
    track.hits.push_back(matched);
    if (static_cast<int>(track.hits.size()) > params_.hits_window) {
      track.hits.pop_front();
    }
    if (!track.confirmed && std::count(track.hits.begin(), track.hits.end(), true) >= params_.min_hits &&
        track.score >= params_.min_score_confirm) {
      track.confirmed = true;
    }
    const double speed = track.velocity().norm();
    if (track.stationary) {
      track.stationary = speed <= params_.stationary_speed + params_.stationary_hysteresis;
    } else {
      track.stationary = speed < params_.stationary_speed;
    }
  }
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [this](const Track& track) { return track.time_since_update > params_.max_age; }),
                tracks_.end());

  for (size_t d = 0; d < detections.size(); ++d) {
    if (!taken[d] && detections[d].score >= params_.min_score_new_track) {
      tracks_.push_back(start(detections[d], stamp));
    }
  }
  return true;
}

void PeopleTracker::predict(Track& track, double dt) const {
  Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
  F(0, 2) = F(1, 3) = dt;
  // White acceleration noise, the same on both axes
  const double q = params_.process_noise_acceleration * params_.process_noise_acceleration;
  Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
  Q(0, 0) = Q(1, 1) = q * dt * dt * dt * dt / 4.0;
  Q(0, 2) = Q(2, 0) = Q(1, 3) = Q(3, 1) = q * dt * dt * dt / 2.0;
  Q(2, 2) = Q(3, 3) = q * dt * dt;
  track.state = F * track.state;
  track.covariance = F * track.covariance * F.transpose() + Q;
}

void PeopleTracker::correct(Track& track, const Detection& detection, double stamp) const {
  Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
  H(0, 0) = H(1, 1) = 1.0;
  const Eigen::Matrix2d R =
      Eigen::Matrix2d::Identity() * params_.measurement_noise_position * params_.measurement_noise_position;
  const Eigen::Vector2d measured = detection.position.head<2>();
  const Eigen::Matrix2d S = H * track.covariance * H.transpose() + R;
  const Eigen::Matrix<double, 4, 2> K = track.covariance * H.transpose() * S.inverse();
  track.state += K * (measured - H * track.state);
  track.covariance = (Eigen::Matrix4d::Identity() - K * H) * track.covariance;

  // Height, size and heading are not filtered, only smoothed
  const double a = params_.shape_smoothing;
  track.z += a * (detection.position.z() - track.z);
  track.dimensions += a * (detection.dimensions - track.dimensions);
  track.yaw = wrapAngle(track.yaw + a * wrapAngle(detection.yaw - track.yaw));

  track.score += (detection.score - track.score) / (track.detections + 1);
  ++track.detections;
  track.time_since_update = 0.0;
  track.last_detection = measured;
  track.last_detection_stamp = stamp;
}

Track PeopleTracker::start(const Detection& detection, double stamp) {
  Track track;
  for (size_t i = 0; i < track.id.size(); i += 8) {
    const uint64_t word = id_generator_();
    for (size_t byte = 0; byte < 8; ++byte) {
      track.id[i + byte] = static_cast<uint8_t>(word >> (8 * byte));
    }
  }
  track.state << detection.position.x(), detection.position.y(), 0.0, 0.0;
  const double position_variance = params_.measurement_noise_position * params_.measurement_noise_position;
  const double speed_variance = params_.initial_speed_std * params_.initial_speed_std;
  track.covariance.diagonal() << position_variance, position_variance, speed_variance, speed_variance;
  track.z = detection.position.z();
  track.dimensions = detection.dimensions;
  track.yaw = detection.yaw;
  track.score = detection.score;
  track.stationary = true;  // no velocity yet
  track.hits.push_back(true);
  track.detections = 1;
  track.last_detection = detection.position.head<2>();
  track.last_detection_stamp = stamp;
  track.confirmed = params_.min_hits <= 1 && track.score >= params_.min_score_confirm;
  return track;
}

}  // namespace people_perception
