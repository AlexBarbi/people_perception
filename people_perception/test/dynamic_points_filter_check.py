#!/usr/bin/env python3
"""Checks people_perception in a running stack: FAST-LIO and what fast_lio_slam_launch.xml starts with it, on a robot
that stands or walks.

Reads --scans consecutive scans of the two FAST-LIO clouds, of their static copies and of /Odometry, prints each check
and exits 1 if any fails:
  receive     all the topics arrive
  registered  every scan of /cloud_registered comes out once on /registered_scan_static, none lost, none twice, same
              header and layout, and
                without --people  (people:=false, a pass-through): the same bytes
                with --people     (people:=true): the same points in the same order, but those taken out
  body        the same for /cloud_registered_body and /cloud_registered_body_static
  stamps      the two FAST-LIO clouds and /Odometry of a scan carry the same stamp (SC-PGO and the filter pair on it)
  points      the two FAST-LIO clouds of a scan have the same number of points (publish.dense_publish_en)
and with --people:
  removed     /perception/dynamic_points holds exactly what is missing from /registered_scan_static
  detector    every scan has its /perception/detector/pointcloud: PointXYZIRC (16 bytes), frame sensor_level
  tracks      every scan has its /perception/tracks_all and /perception/tracks, in the frame of /cloud_registered
For information: the scan rate, the point layout, how many points a scan loses, how many people are tracked, and how
long after its input a static cloud arrives here.
"""
import argparse
import statistics
import sys
import time
from collections import Counter

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField

PAIRS = {'registered': ('/cloud_registered', '/registered_scan_static'),
         'body': ('/cloud_registered_body', '/cloud_registered_body_static')}
ODOMETRY = '/Odometry'
REMOVED = '/perception/dynamic_points'
DETECTOR_CLOUD = '/perception/detector/pointcloud'
TRACKS = ('/perception/tracks_all', '/perception/tracks')
XYZIRC = [('x', 0, PointField.FLOAT32), ('y', 4, PointField.FLOAT32), ('z', 8, PointField.FLOAT32),
          ('intensity', 12, PointField.UINT8), ('return_type', 13, PointField.UINT8),
          ('channel', 14, PointField.UINT16)]


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def layout(msg):
    """Everything a subscriber sees of a cloud but its stamp, its size and its points."""
    fields = tuple((f.name, f.offset, f.datatype, f.count) for f in msg.fields)
    return msg.header.frame_id, fields, msg.is_bigendian, msg.point_step, msg.is_dense


def rows(msg):
    """The points of a cloud, one row of bytes each."""
    count = msg.height * msg.width
    return np.frombuffer(msg.data, dtype=np.uint8, count=count * msg.point_step).reshape(count, msg.point_step).copy()


def is_what_is_left(kept, cloud):
    """True if the rows of `kept` are rows of `cloud` in the same order. Also returns the rows of `cloud` left out."""
    weights = np.random.default_rng(0).integers(1, 2**62, cloud.shape[1], dtype=np.uint64)
    kept_keys = kept.astype(np.uint64) @ weights
    in_kept = np.isin(cloud.astype(np.uint64) @ weights, kept_keys)
    return np.array_equal(cloud[in_kept], kept), cloud[~in_kept]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scans', type=int, default=50)
    parser.add_argument('--timeout', type=float, default=60.0, help='wall time to wait for --scans scans [s]')
    parser.add_argument('--people', action='store_true', help='the stack runs with people:=true')
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node('dynamic_points_filter_check')
    cloud_topics = [topic for pair in PAIRS.values() for topic in pair] + ([REMOVED] if args.people else [])
    other_topics = [ODOMETRY] + ([DETECTOR_CLOUD, *TRACKS] if args.people else [])
    seen = {topic: {} for topic in cloud_topics + other_topics}  # topic -> stamp [ns] -> (what is kept, receive time)
    twice = Counter()

    def keep(topic, of):
        def callback(msg):
            stamp = stamp_ns(msg)
            twice[topic] += stamp in seen[topic]
            seen[topic][stamp] = (of(msg), time.monotonic())
        return callback

    # Reliable: a best-effort subscription loses most of the ~1 MB clouds
    for topic in cloud_topics:
        node.create_subscription(PointCloud2, topic, keep(topic, lambda msg: (layout(msg), rows(msg))), 20)
    node.create_subscription(Odometry, ODOMETRY, keep(ODOMETRY, lambda msg: None), 20)
    if args.people:
        from autoware_perception_msgs.msg import TrackedObjects
        node.create_subscription(PointCloud2, DETECTOR_CLOUD,
                                 keep(DETECTOR_CLOUD, lambda msg: (layout(msg), msg.height * msg.width)), 20)
        for topic in TRACKS:
            node.create_subscription(
                TrackedObjects, topic,
                keep(topic, lambda msg: (msg.header.frame_id, len(msg.objects))), 20)

    reference = PAIRS['registered'][0]
    deadline = time.monotonic() + args.timeout
    start = None  # first stamp from which every topic is being received
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if start is None and all(seen.values()):
            start = max(min(stamps) for stamps in seen.values())
        if start is not None and sum(stamp >= start for stamp in seen[reference]) >= args.scans:
            break
    inputs = {name: sorted(seen[pair[0]]) for name, pair in PAIRS.items()}  # the scans that are checked
    end = time.monotonic() + 1.5  # their static copies, tracks and odometry may still be on the way
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

    silent = [topic for topic, stamps in seen.items() if not stamps]
    if silent:
        print(f'FAIL receive: nothing on {", ".join(silent)} in {args.timeout:.0f} s')
        return 1
    scans = [stamp for stamp in inputs['registered'] if stamp >= start]
    if len(scans) < 2:
        print(f'FAIL receive: {len(scans)} scans on all the topics together in {args.timeout:.0f} s')
        return 1
    rate = (len(scans) - 1) / ((scans[-1] - scans[0]) * 1e-9)
    results = [('receive', len(scans) >= args.scans,
                f'{len(scans)} scans from {scans[0] * 1e-9:.3f} to {scans[-1] * 1e-9:.3f} s, {rate:.1f} Hz')]

    delays, taken_out = [], {}
    for name, (input_topic, output_topic) in PAIRS.items():
        checked = [stamp for stamp in inputs[name] if stamp >= start]
        lost = [stamp for stamp in checked if stamp not in seen[output_topic]]
        unknown = [stamp for stamp in seen[output_topic] if start <= stamp <= checked[-1] and stamp not in seen[input_topic]]
        changed, removed_points = [], []
        for stamp in checked:
            if stamp not in seen[output_topic]:
                continue
            (cloud_layout, cloud), (out_layout, out) = seen[input_topic][stamp][0], seen[output_topic][stamp][0]
            if args.people:
                same, left_out = is_what_is_left(out, cloud)
                taken_out[(name, stamp)] = left_out
                removed_points.append(len(left_out))
            else:
                same = np.array_equal(cloud, out)
            if not same or out_layout != cloud_layout:
                changed.append(stamp)
            delays.append(seen[output_topic][stamp][1] - seen[input_topic][stamp][1])
        (frame, _, _, point_step, _), last = seen[input_topic][checked[-1]][0]
        detail = (f'{output_topic} = {input_topic} on {len(checked) - len(lost) - len(changed)} of {len(checked)} scans: '
                  f'{len(lost)} lost, {len(changed)} changed, {len(unknown)} with an unknown stamp, '
                  f'{twice[output_topic]} twice (frame {frame}, {len(last)} points, point_step {point_step})')
        if args.people and removed_points:
            detail += f'; points taken out of a scan: {statistics.mean(removed_points):.0f} mean, {max(removed_points)} max'
        results.append((name, not (lost or changed or unknown or twice[output_topic]), detail))

    others = {topic: [stamp for stamp in scans if stamp not in seen[topic]] for topic in (PAIRS['body'][0], ODOMETRY)}
    results.append(('stamps', not any(others.values()),
                    f'of the {len(scans)} {reference} stamps, {len(others[PAIRS["body"][0]])} are not on '
                    f'{PAIRS["body"][0]} and {len(others[ODOMETRY])} not on {ODOMETRY}'))

    both = [stamp for stamp in scans if stamp in seen[PAIRS['body'][0]]]
    unequal = [stamp for stamp in both
               if len(seen[reference][stamp][0][1]) != len(seen[PAIRS['body'][0]][stamp][0][1])]
    results.append(('points', bool(both) and not unequal,
                    f'{len(unequal)} of {len(both)} scans with a different number of points in the two clouds'))

    if args.people:
        compared = [stamp for stamp in scans if ('registered', stamp) in taken_out and stamp in seen[REMOVED]]
        wrong = [stamp for stamp in compared
                 if not np.array_equal(seen[REMOVED][stamp][0][1], taken_out[('registered', stamp)])]
        results.append(('removed', len(compared) == len(scans) and not wrong,
                        f'{REMOVED} is what is missing from {PAIRS["registered"][1]} on {len(compared) - len(wrong)} '
                        f'of {len(scans)} scans'))

        missing = [stamp for stamp in scans if stamp not in seen[DETECTOR_CLOUD]]
        (frame, fields, _, point_step, _), points = seen[DETECTOR_CLOUD][max(seen[DETECTOR_CLOUD])][0]
        results.append(('detector', not missing and list(fields) == [(*f, 1) for f in XYZIRC] and point_step == 16,
                        f'{DETECTOR_CLOUD} on {len(scans) - len(missing)} of {len(scans)} scans (frame {frame}, '
                        f'{points} points, point_step {point_step}, fields {" ".join(f[0] for f in fields)})'))

        cloud_frame = seen[reference][scans[-1]][0][0][0]
        unanswered = {topic: [stamp for stamp in scans if stamp not in seen[topic]] for topic in TRACKS}
        frames = {seen[topic][stamp][0][0] for topic in TRACKS for stamp in scans if stamp in seen[topic]}
        people = [seen[TRACKS[1]][stamp][0][1] for stamp in scans if stamp in seen[TRACKS[1]]]
        results.append(('tracks', not any(unanswered.values()) and frames == {cloud_frame},
                        f'{len(unanswered[TRACKS[0]])} scans without {TRACKS[0]}, {len(unanswered[TRACKS[1]])} without '
                        f'{TRACKS[1]}; frame {", ".join(sorted(frames))}; confirmed people in a scan: '
                        f'{min(people, default=0)} to {max(people, default=0)}'))

    for name, ok, detail in results:
        print(f'{"PASS" if ok else "FAIL"} {name}: {detail}')
    fields = seen[reference][scans[-1]][0][0][1]
    print('INFO layout: ' + ' '.join(f'{field[0]}@{field[1]}' for field in fields))
    if delays:
        print(f'INFO delay: a static cloud arrives {statistics.median(delays) * 1e3:.1f} ms (median), '
              f'{max(delays) * 1e3:.1f} ms (max) after its input')
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == '__main__':
    sys.exit(main())
