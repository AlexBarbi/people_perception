#include "people_perception/DynamicPointsFilter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

namespace people_perception {

namespace {

int64_t stampOf(const std_msgs::msg::Header& header) { return rclcpp::Time(header.stamp).nanoseconds(); }

// Largest variance along any direction of the plane, from the x y block of a 6 x 6 covariance.
double largestVariance(const std::array<double, 36>& covariance) {
  const double a = covariance[0], b = covariance[1], d = covariance[7];
  return std::max(0.0, (a + d) / 2.0 + std::sqrt((a - d) * (a - d) / 4.0 + b * b));
}

// An empty cloud with the header and the layout of `like`.
std::unique_ptr<sensor_msgs::msg::PointCloud2> emptyLike(const sensor_msgs::msg::PointCloud2& like) {
  auto cloud = std::make_unique<sensor_msgs::msg::PointCloud2>();
  cloud->header = like.header;
  cloud->fields = like.fields;
  cloud->is_bigendian = like.is_bigendian;
  cloud->point_step = like.point_step;
  cloud->is_dense = like.is_dense;
  cloud->height = 1;
  return cloud;
}

}  // namespace

DynamicPointsFilter::DynamicPointsFilter(const rclcpp::NodeOptions& options)
    : Node("dynamic_points_filter", options), qos_(1), diagnostics_(this) {
  enabled_ = declare_parameter<bool>("enabled");
  // Reliable on both sides, as FAST-LIO publishes: its clouds are about 1 MB and best effort loses them to fragment
  // drops
  qos_ = rclcpp::QoS(static_cast<size_t>(declare_parameter<int>("qos_depth")));
  for (const char* name : {"registered_cloud", "body_cloud"}) {
    Stream stream;
    stream.input_topic = declare_parameter<std::string>(std::string(name) + ".input_topic");
    stream.output_topic = declare_parameter<std::string>(std::string(name) + ".output_topic");
    stream.body = std::string(name) == "body_cloud";
    streams_.push_back(std::move(stream));
  }

  // Read with enabled false too: a value missing from the yaml shows now, not the day the filtering is switched on
  odometry_topic_ = declare_parameter<std::string>("odometry_topic");
  tracks_topic_ = declare_parameter<std::string>("tracks_topic");
  removed_points_topic_ = declare_parameter<std::string>("removed_points_topic");
  publish_removed_points_ = declare_parameter<bool>("publish_removed_points");
  margin_xy_ = declare_parameter<double>("margin_xy");
  min_radius_ = declare_parameter<double>("min_radius");
  margin_z_down_ = declare_parameter<double>("margin_z_down");
  margin_z_up_ = declare_parameter<double>("margin_z_up");
  position_sigma_factor_ = declare_parameter<double>("position_sigma_factor");
  max_wait_ = declare_parameter<double>("max_wait");
  max_track_staleness_ = declare_parameter<double>("max_track_staleness");
  queue_size_ = static_cast<size_t>(declare_parameter<int>("queue_size"));
  const std::string on_timeout = declare_parameter<std::string>("on_timeout");
  if (on_timeout == "last_tracks") {
    on_timeout_ = OnTimeout::kLastTracks;
  } else if (on_timeout == "passthrough") {
    on_timeout_ = OnTimeout::kPassthrough;
  } else if (on_timeout == "drop") {
    on_timeout_ = OnTimeout::kDrop;
  } else {
    throw std::invalid_argument("on_timeout: '" + on_timeout + "' is not last_tracks, passthrough or drop");
  }

  diagnostics_.setHardwareID("none");
  diagnostics_.add("scans", this, &DynamicPointsFilter::diagnose);

  // Nothing goes out before this node knows it is the only filter, see start()
  const std::chrono::duration<double> wait(declare_parameter<double>("duplicate_check_wait"));
  start_timer_ = create_wall_timer(wait, [this] {
    start_timer_->cancel();
    start();
  });
}

void DynamicPointsFilter::start() {
  // A second filter would hand every scan to SC-PGO and TARE twice. SC-PGO then pairs the second copy with the
  // odometry of the next scan and falls one scan further behind on every scan: wrong keyframe poses, for good.
  for (const Stream& stream : streams_) {
    if (count_publishers(stream.output_topic) > 0) {
      throw std::runtime_error(stream.output_topic +
                               " is already published: another dynamic_points_filter is running (fast_lio "
                               "fast_lio_slam_launch.xml starts one). This one stops, it would double every scan");
    }
  }
  for (Stream& stream : streams_) {
    stream.publisher = create_publisher<PointCloud2>(stream.output_topic, qos_);
    stream.subscription = create_subscription<PointCloud2>(
        stream.input_topic, qos_, [this, &stream](PointCloud2::UniquePtr cloud) { onCloud(stream, std::move(cloud)); });
    RCLCPP_INFO(get_logger(), "%s -> %s, %s", stream.input_topic.c_str(), stream.output_topic.c_str(),
                enabled_ ? "without the tracked people" : "unchanged");
  }
  if (!enabled_) {
    return;
  }
  if (publish_removed_points_) {
    removed_publisher_ = create_publisher<PointCloud2>(removed_points_topic_, qos_);
  }
  tracks_subscription_ = create_subscription<TrackedObjects>(
      tracks_topic_, qos_, [this](TrackedObjects::ConstSharedPtr tracks) { onTracks(std::move(tracks)); });
  odometry_subscription_ = create_subscription<Odometry>(
      odometry_topic_, qos_, [this](Odometry::ConstSharedPtr odometry) { onOdometry(*odometry); });
  // max_wait is checked ten times within it
  overdue_timer_ = create_wall_timer(std::chrono::duration<double>(max_wait_ / 10.0), [this] {
    for (Stream& stream : streams_) {
      release(stream);
    }
  });
}

void DynamicPointsFilter::onCloud(Stream& stream, PointCloud2::UniquePtr cloud) {
  if (!enabled_) {
    stream.publisher->publish(std::move(cloud));
    return;
  }
  const int64_t stamp = stampOf(cloud->header);
  stream.waiting[stamp] = Waiting{std::move(cloud), std::chrono::steady_clock::now()};
  release(stream);
}

void DynamicPointsFilter::onTracks(TrackedObjects::ConstSharedPtr tracks) {
  const int64_t stamp = stampOf(tracks->header);
  const int64_t staleness = static_cast<int64_t>(max_track_staleness_ * 1e9);
  if (!tracks_.empty() && stamp < tracks_.rbegin()->first - staleness) {
    tracks_.clear();  // the clock went back (a bag started again)
  }
  tracks_[stamp] = std::move(tracks);
  // Only those a late cloud could still use
  while (tracks_.begin()->first < tracks_.rbegin()->first - staleness) {
    tracks_.erase(tracks_.begin());
  }
  for (Stream& stream : streams_) {
    release(stream);
  }
}

void DynamicPointsFilter::onOdometry(const Odometry& odometry) {
  const int64_t stamp = stampOf(odometry.header);
  if (!poses_.empty() && stamp < poses_.rbegin()->first - static_cast<int64_t>(max_track_staleness_ * 1e9)) {
    poses_.clear();  // the clock went back
  }
  const auto& p = odometry.pose.pose.position;
  const auto& q = odometry.pose.pose.orientation;
  poses_[stamp] = Pose{odometry.header.frame_id,
                       Eigen::Translation3d(p.x, p.y, p.z) * Eigen::Quaterniond(q.w, q.x, q.y, q.z)};
  // A body cloud waits queue_size scans at most: twice that is every pose one can still ask for
  while (poses_.size() > 2 * queue_size_) {
    poses_.erase(poses_.begin());
  }
  for (Stream& stream : streams_) {
    if (stream.body) {
      release(stream);
    }
  }
}

void DynamicPointsFilter::release(Stream& stream) {
  const auto now = std::chrono::steady_clock::now();
  while (!stream.waiting.empty()) {
    const auto oldest = stream.waiting.begin();
    const int64_t stamp = oldest->first;
    const auto tracks = tracks_.find(stamp);
    const bool has_tracks = tracks != tracks_.end();
    // The tracker answers the scans in order: once it has answered a later one, this one will not get its tracks
    const bool skipped = !has_tracks && !tracks_.empty() && tracks_.rbegin()->first > stamp;
    // The odometry of a scan can come after the tracks of the next one: a body cloud waits for it all the same
    const bool has_pose = !stream.body || poses_.count(stamp) > 0;
    const bool overdue = std::chrono::duration<double>(now - oldest->second.since).count() > max_wait_ ||
                         stream.waiting.size() > queue_size_;
    if (!overdue && !(has_pose && (has_tracks || skipped))) {
      return;  // the oldest still waits, and nothing goes out before it
    }
    Waiting cloud = std::move(oldest->second);
    stream.waiting.erase(oldest);
    if (has_tracks && has_pose) {
      filterAndPublish(stream, std::move(cloud), *tracks->second, 0.0);
    } else {
      closeWithoutItsTracks(stream, std::move(cloud));
    }
  }
}

void DynamicPointsFilter::closeWithoutItsTracks(Stream& stream, Waiting waiting) {
  ++without_tracks_;
  if (on_timeout_ == OnTimeout::kDrop) {
    ++dropped_;
    return;
  }
  if (on_timeout_ == OnTimeout::kLastTracks && !tracks_.empty()) {
    const int64_t stamp = stampOf(waiting.cloud->header);
    // The tracks of the scan closest in time, before or after
    const auto after = tracks_.lower_bound(stamp);
    auto closest = after;
    if (after == tracks_.end() ||
        (after != tracks_.begin() && stamp - std::prev(after)->first < after->first - stamp)) {
      closest = std::prev(after);
    }
    const double dt = static_cast<double>(stamp - closest->first) * 1e-9;
    if (std::abs(dt) <= max_track_staleness_ && (!stream.body || poses_.count(stamp) > 0)) {
      filterAndPublish(stream, std::move(waiting), *closest->second, dt);
      return;
    }
  }
  publishUnfiltered(stream, std::move(waiting));
}

void DynamicPointsFilter::filterAndPublish(Stream& stream, Waiting waiting, const TrackedObjects& tracks, double dt) {
  Eigen::Isometry3d tracks_from_cloud = Eigen::Isometry3d::Identity();
  std::string frame = waiting.cloud->header.frame_id;
  if (stream.body) {
    const Pose& pose = poses_.at(stampOf(waiting.cloud->header));
    tracks_from_cloud = pose.frame_from_body;
    frame = pose.frame;
  }
  if (frame != tracks.header.frame_id) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                          "The tracks are in %s, %s is in %s: not filtered. The tracker must work in the frame of "
                          "the FAST-LIO odometry",
                          tracks.header.frame_id.c_str(), stream.input_topic.c_str(), frame.c_str());
    publishUnfiltered(stream, std::move(waiting));
    return;
  }
  const std::vector<Cylinder> cylinders = cylindersOf(tracks, dt);
  if (cylinders.empty()) {
    ++filtered_;
    publishUnfiltered(stream, std::move(waiting));  // nobody around: the very cloud that came in
    return;
  }
  auto kept = std::make_unique<PointCloud2>();
  std::unique_ptr<PointCloud2> removed;
  if (!stream.body && removed_publisher_) {
    removed = std::make_unique<PointCloud2>();
  }
  const auto count = removeInsideCylinders(*waiting.cloud, cylinders, tracks_from_cloud, *kept, removed.get());
  if (!count) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                          "%s without FLOAT32 x y z fields or with short data: not filtered", stream.input_topic.c_str());
    publishUnfiltered(stream, std::move(waiting));
    return;
  }
  longest_wait_ = std::max(
      longest_wait_, std::chrono::duration<double>(std::chrono::steady_clock::now() - waiting.since).count());
  ++filtered_;
  points_removed_ += *count;
  stream.publisher->publish(std::move(kept));
  if (removed) {
    removed_publisher_->publish(std::move(removed));
  }
}

void DynamicPointsFilter::publishUnfiltered(Stream& stream, Waiting waiting) {
  longest_wait_ = std::max(
      longest_wait_, std::chrono::duration<double>(std::chrono::steady_clock::now() - waiting.since).count());
  if (!stream.body && removed_publisher_) {
    removed_publisher_->publish(emptyLike(*waiting.cloud));  // nothing taken out of this scan
  }
  stream.publisher->publish(std::move(waiting.cloud));
}

std::vector<Cylinder> DynamicPointsFilter::cylindersOf(const TrackedObjects& tracks, double dt) const {
  std::vector<Cylinder> cylinders;
  cylinders.reserve(tracks.objects.size());
  for (const auto& object : tracks.objects) {
    const auto& pose = object.kinematics.pose_with_covariance;
    const auto& twist = object.kinematics.twist_with_covariance;
    const auto& size = object.shape.dimensions;
    // Where the person may be: around the track, moved on by dt, as far as the tracker is unsure of position and,
    // over dt, of velocity
    const double sigma = std::sqrt(largestVariance(pose.covariance) + dt * dt * largestVariance(twist.covariance));
    Cylinder cylinder;
    cylinder.x = pose.pose.position.x + twist.twist.linear.x * dt;
    cylinder.y = pose.pose.position.y + twist.twist.linear.y * dt;
    cylinder.radius = std::max(std::max(size.x, size.y) / 2.0 + margin_xy_, min_radius_) + position_sigma_factor_ * sigma;
    cylinder.z_min = pose.pose.position.z - size.z / 2.0 - margin_z_down_;
    cylinder.z_max = pose.pose.position.z + size.z / 2.0 + margin_z_up_;
    cylinders.push_back(cylinder);
  }
  return cylinders;
}

void DynamicPointsFilter::diagnose(diagnostic_updater::DiagnosticStatusWrapper& status) {
  using diagnostic_msgs::msg::DiagnosticStatus;
  if (!enabled_) {
    status.summary(DiagnosticStatus::OK, "pass-through: the clouds are not filtered");
    return;
  }
  if (without_tracks_ > 0) {
    status.summaryf(DiagnosticStatus::WARN,
                    "%zu clouds went without the tracks or the odometry of their scan (detector late or down)",
                    without_tracks_);
  } else {
    status.summary(DiagnosticStatus::OK, "filtering");
  }
  status.add("clouds filtered", filtered_);
  status.add("clouds without the tracks or the odometry of their scan", without_tracks_);
  status.add("clouds dropped", dropped_);
  status.add("points removed", points_removed_);
  status.addf("longest wait [ms]", "%.1f", longest_wait_ * 1e3);
  filtered_ = without_tracks_ = dropped_ = points_removed_ = 0;
  longest_wait_ = 0.0;
}

}  // namespace people_perception
