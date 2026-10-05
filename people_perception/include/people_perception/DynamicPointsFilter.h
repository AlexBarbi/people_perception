#pragma once

// ROS 2 node: block B6 of docs/perception_pipeline.md. It stands between FAST-LIO and what accumulates its scans over
// time, and takes the points of the tracked people out of them:
//   /cloud_registered       (camera_init)  ->  /registered_scan_static         TARE coverage
//   /cloud_registered_body  (rslidar_lio)  ->  /cloud_registered_body_static   SC-PGO keyframes
//   /perception/dynamic_points             what was taken out of /cloud_registered, to look at
// The outputs have the fields, the point_step, the frame and the stamp of the inputs: TARE and SC-PGO cannot tell
// them from the FAST-LIO clouds, but for the missing points. Topics and every other parameter come from
// config/dynamic_points_filter.yaml: there are no defaults here, a missing one stops the node.
//
// enabled: false   every cloud goes out as it came in, at once.
// enabled: true    a cloud waits for the tracks of its scan (people_tracker answers every scan, with the stamp of the
//                  scan). Around each track, tentative and coasting ones too, a vertical cylinder: the box plus
//                  margins, plus position_sigma_factor times the position uncertainty the tracker reports, which
//                  grows while a track coasts. The points inside go. The body cloud is tested through the odometry
//                  of its scan.
//                  A cloud whose tracks do not come within max_wait, or cannot come any more (the tracks of a later
//                  scan are here, the queue is full), goes by on_timeout: last_tracks (the tracks of the closest
//                  scan, no farther than max_track_staleness, moved on with their velocity; unfiltered without any),
//                  passthrough (unfiltered) or drop. /diagnostics tells how many did.
//                  The clouds of a topic go out in the order of the scans, whatever each one waits for: SC-PGO pairs
//                  each cloud with the next odometry in its queue, one cloud out of turn shifts every later pair.
//
// With the detector down the clouds go out unfiltered: this node is not what avoids collisions. What does
// (terrainAnalysis, terrainAnalysisExt) must never read its outputs.
//
// One filter only: the node starts duplicate_check_wait after its launch, and stops instead if an output is already
// published by then.

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "people_perception/CloudFilter.h"

namespace people_perception {

class DynamicPointsFilter : public rclcpp::Node {
 public:
  explicit DynamicPointsFilter(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  using PointCloud2 = sensor_msgs::msg::PointCloud2;
  using Odometry = nav_msgs::msg::Odometry;
  using TrackedObjects = autoware_perception_msgs::msg::TrackedObjects;

  struct Waiting {
    PointCloud2::UniquePtr cloud;
    std::chrono::steady_clock::time_point since;
  };

  // One of the two clouds of FAST-LIO and its static copy.
  struct Stream {
    std::string input_topic;
    std::string output_topic;
    bool body = false;  // in the frame of the robot: tested through the odometry of the scan
    rclcpp::Publisher<PointCloud2>::SharedPtr publisher;
    rclcpp::Subscription<PointCloud2>::SharedPtr subscription;
    std::map<int64_t, Waiting> waiting;  // the clouds whose tracks have not come, by stamp [ns]
  };

  struct Pose {
    std::string frame;
    Eigen::Isometry3d frame_from_body;
  };

  enum class OnTimeout { kLastTracks, kPassthrough, kDrop };

  /// Start republishing, unless another node already publishes an output.
  void start();

  void onCloud(Stream& stream, PointCloud2::UniquePtr cloud);
  void onTracks(TrackedObjects::ConstSharedPtr tracks);
  void onOdometry(const Odometry& odometry);

  /// Let out, in the order of the scans, every waiting cloud of `stream` that has what it needs or cannot have it.
  void release(Stream& stream);
  /// A cloud that goes without the tracks or the odometry of its scan, by on_timeout.
  void closeWithoutItsTracks(Stream& stream, Waiting waiting);
  /// Publish `cloud` without the people of `tracks`, which are `dt` seconds older than the cloud.
  void filterAndPublish(Stream& stream, Waiting waiting, const TrackedObjects& tracks, double dt);
  void publishUnfiltered(Stream& stream, Waiting waiting);
  std::vector<Cylinder> cylindersOf(const TrackedObjects& tracks, double dt) const;
  void diagnose(diagnostic_updater::DiagnosticStatusWrapper& status);

  bool enabled_ = false;
  rclcpp::QoS qos_;
  std::vector<Stream> streams_;
  rclcpp::TimerBase::SharedPtr start_timer_;

  // enabled only
  std::string odometry_topic_, tracks_topic_, removed_points_topic_;
  bool publish_removed_points_ = false;
  double margin_xy_ = 0.0, min_radius_ = 0.0, margin_z_down_ = 0.0, margin_z_up_ = 0.0, position_sigma_factor_ = 0.0;
  double max_wait_ = 0.0, max_track_staleness_ = 0.0;
  OnTimeout on_timeout_ = OnTimeout::kLastTracks;
  size_t queue_size_ = 0;
  std::map<int64_t, TrackedObjects::ConstSharedPtr> tracks_;  // of the last scans, by stamp [ns]
  std::map<int64_t, Pose> poses_;                             // odometry of the last scans, by stamp [ns]
  rclcpp::Publisher<PointCloud2>::SharedPtr removed_publisher_;
  rclcpp::Subscription<TrackedObjects>::SharedPtr tracks_subscription_;
  rclcpp::Subscription<Odometry>::SharedPtr odometry_subscription_;
  rclcpp::TimerBase::SharedPtr overdue_timer_;

  // Since the last /diagnostics report
  diagnostic_updater::Updater diagnostics_;
  size_t filtered_ = 0, without_tracks_ = 0, dropped_ = 0, points_removed_ = 0;
  double longest_wait_ = 0.0;
};

}  // namespace people_perception
