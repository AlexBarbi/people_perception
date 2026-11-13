#!/usr/bin/env python3
"""LaserScan -> DR-SPAAM / ONNX Runtime CPU -> tracker-compatible DetectedObjects."""
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from autoware_perception_msgs.msg import (
    DetectedObject, DetectedObjectKinematics, DetectedObjects, ObjectClassification, Shape,
)

from dr_spaam_inference import OnnxDetector, decode


def to_objects(header, xy, scores, width, depth, height, ground_z):
    """2D detector supplies nominal body boxes for tracking and cloud removal."""
    out = DetectedObjects()
    out.header = header
    for (x, y), score in zip(xy, scores):
        obj = DetectedObject()
        obj.existence_probability = float(score)
        obj.classification = [ObjectClassification(
            label=ObjectClassification.PEDESTRIAN, probability=1.0)]
        pose = obj.kinematics.pose_with_covariance.pose
        pose.position.x, pose.position.y = float(x), float(y)
        pose.position.z = ground_z + height / 2
        pose.orientation.w = 1.0
        obj.kinematics.orientation_availability = DetectedObjectKinematics.UNAVAILABLE
        obj.shape.type = Shape.BOUNDING_BOX
        obj.shape.dimensions.x, obj.shape.dimensions.y, obj.shape.dimensions.z = depth, width, height
        out.objects.append(obj)
    return out


class DrSpaamCpu(Node):
    def __init__(self):
        super().__init__('dr_spaam_cpu')
        defaults = dict(model_path='', scan_topic='/perception/scan',
                        objects_topic='/perception/detector/objects', expected_frame='base_link',
                        confidence_threshold=0.5, nms_distance=0.5, stride=1, num_threads=4,
                        range_min=0.3, range_max=15.0, state_timeout=0.5,
                        person_width=0.6, person_depth=0.6, person_height=1.7, ground_z=0.0,
                        latency_budget_ms=15.0)
        self.p = {key: self.declare_parameter(key, value).value for key, value in defaults.items()}
        p = self.p
        if not 0 <= p['confidence_threshold'] <= 1:
            raise ValueError('confidence_threshold must be in [0, 1]')
        positive = ('nms_distance', 'range_min', 'range_max', 'state_timeout', 'person_width',
                    'person_depth', 'person_height', 'latency_budget_ms', 'stride', 'num_threads')
        if any(not math.isfinite(p[key]) or p[key] <= 0 for key in positive):
            raise ValueError('Distances, sizes, timeouts, thread count and stride must be positive')
        if p['range_max'] <= p['range_min'] or p['range_max'] > 29.99:
            raise ValueError('Require range_min < range_max <= 29.99 m')
        if not p['expected_frame'] or not math.isfinite(p['ground_z']):
            raise ValueError('expected_frame and finite ground_z are required')
        path = Path(p['model_path']).expanduser()
        if not path.is_file():
            raise ValueError(f'DR-SPAAM ONNX missing: {path}. Run export_dr_spaam_onnx.py first.')
        self.detector = OnnxDetector(str(path), p['num_threads'], p['stride'])
        self.grid = None
        self.last_stamp = None
        self.publisher = self.create_publisher(DetectedObjects, p['objects_topic'], QoSProfile(depth=1))
        # Keep only the newest waiting scan while inference runs in the single-threaded executor.
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(LaserScan, p['scan_topic'], self.on_scan, qos)
        self.get_logger().info(f'DR-SPAAM CPU: {path}, stride={p["stride"]}, threads={p["num_threads"]}')

    def on_scan(self, msg):
        start = time.perf_counter()
        p = self.p
        if msg.header.frame_id != p['expected_frame']:
            self.get_logger().error(f'Expected scan frame {p["expected_frame"]}, got {msg.header.frame_id}',
                                    throttle_duration_sec=5.0)
            self.detector.reset()
            return
        if (len(msg.ranges) < 2 or not math.isfinite(msg.angle_increment) or msg.angle_increment <= 0 or
                not math.isfinite(msg.angle_min) or not math.isfinite(msg.range_min) or
                not math.isfinite(msg.range_max) or msg.range_max <= msg.range_min):
            self.get_logger().error('Invalid LaserScan geometry', throttle_duration_sec=5.0)
            self.detector.reset()
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        grid = (msg.header.frame_id, len(msg.ranges), msg.angle_min, msg.angle_increment,
                msg.range_min, msg.range_max)
        if (grid != self.grid or self.last_stamp is None or stamp <= self.last_stamp or
                stamp - self.last_stamp > p['state_timeout']):
            self.detector.reset()
        self.grid, self.last_stamp = grid, stamp
        scan = np.asarray(msg.ranges, dtype=np.float32)
        valid = (np.isfinite(scan) & (scan >= max(msg.range_min, p['range_min'])) &
                 (scan <= min(msg.range_max, p['range_max'])))
        # Far padding matches training. Never let invalid beams cast person votes.
        scan = np.where(valid, scan, 29.99).astype(np.float32)
        try:
            if np.any(valid[::p['stride']]):
                logits, offsets = self.detector.infer(scan, msg.angle_increment)
                angles = msg.angle_min + np.arange(0, len(scan), p['stride']) * msg.angle_increment
                xy, scores = decode(scan[::p['stride']], angles, valid[::p['stride']], logits, offsets,
                                    p['confidence_threshold'], p['nms_distance'],
                                    p['range_min'], p['range_max'])
            else:
                self.detector.reset()
                xy, scores = [], []
        except Exception as error:
            # Inference failure is not a valid empty detection; downstream sees missing perception.
            self.detector.reset()
            self.get_logger().error(f'DR-SPAAM inference failed: {error}', throttle_duration_sec=5.0)
            return
        self.publisher.publish(to_objects(msg.header, xy, scores, p['person_width'], p['person_depth'],
                                          p['person_height'], p['ground_z']))
        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > p['latency_budget_ms']:
            self.get_logger().warning(f'DR-SPAAM callback {elapsed:.1f} ms > {p["latency_budget_ms"]} ms budget',
                                      throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DrSpaamCpu()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
