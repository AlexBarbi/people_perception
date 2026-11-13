"""Geometric, boundary and ROS contract checks for the 2D detector."""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from dr_spaam_inference import cutouts, decode


def test_cutout_normalization_area_sampling_and_padding():
    # A constant far scan fills each cutout with zeros except out-of-sector padding.
    far = cutouts(np.full(101, 5, dtype=np.float32), 0.00785)
    assert far.shape == (1, 101, 1, 56)
    assert far.dtype == np.float32 and far.flags.c_contiguous
    np.testing.assert_array_equal(far[0, 50], 0)
    assert far[0, 0, 0, 0] == 1
    # Close scan invokes area resampling; a constant surface still normalizes to zero.
    close = cutouts(np.full(801, 0.3, dtype=np.float32), 0.00785, stride=2)
    np.testing.assert_allclose(close[0, 200], 0, atol=1e-6)


def test_offsets_rotate_into_ros_axes_and_nms_keeps_highest_score():
    xy, score = decode(np.array([2., 2., 2.]), np.array([0., 0., math.pi / 2]),
                       np.ones(3, dtype=bool), np.array([1., 3., 2.]),
                       np.array([[0., 0.], [0.1, 0.], [0.4, 0.5]]))
    np.testing.assert_allclose(xy, [[2., 0.1], [-0.4, 2.5]], atol=1e-7)
    assert score[0] > score[1] > 0.5


def test_invalid_beams_low_scores_and_nonfinite_votes_cannot_make_people():
    xy, scores = decode(np.full(4, 2.), np.zeros(4), np.array([False, True, True, True]),
                        np.array([10., -1., 10., 10.]),
                        np.array([[0., 0.], [0., 0.], [np.nan, 0.], [0., 20.]]))
    assert xy.shape == (0, 2) and scores.size == 0


def load_node():
    pytest.importorskip('rclpy')
    pytest.importorskip('autoware_perception_msgs')
    spec = importlib.util.spec_from_file_location('dr_spaam_cpu', SCRIPTS / 'dr_spaam_cpu.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracker_message_preserves_header_class_score_and_full_body_box():
    node = load_node()
    from std_msgs.msg import Header
    from autoware_perception_msgs.msg import ObjectClassification
    header = Header(frame_id='base_link')
    header.stamp.sec, header.stamp.nanosec = 12, 345678901
    out = node.to_objects(header, [[2., -1.]], [0.8], 0.6, 0.6, 1.7, 0.0)
    assert out.header == header
    obj = out.objects[0]
    assert obj.classification[0].label == ObjectClassification.PEDESTRIAN
    assert obj.existence_probability == pytest.approx(0.8)
    pose = obj.kinematics.pose_with_covariance.pose
    assert (pose.position.x, pose.position.y, pose.position.z) == (2., -1., 0.85)
    assert pose.orientation.w == 1.0
    assert obj.shape.dimensions.z == 1.7
    assert not node.to_objects(header, [], [], 0.6, 0.6, 1.7, 0.).objects


def test_floor_tf_uses_floor_height_and_yaw_preserving_odometry_stamp():
    pytest.importorskip('rclpy')
    from nav_msgs.msg import Odometry
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('perception_floor_tf', SCRIPTS / 'perception_floor_tf.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    odometry = Odometry()
    odometry.header.frame_id = 'odom'
    odometry.header.stamp.sec = 12
    odometry.header.stamp.nanosec = 345678901
    pose = odometry.pose.pose
    pose.position.x, pose.position.y, pose.position.z = 3., -2., 0.39
    roll, pitch, yaw = 0.1, -0.15, 0.8
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    pose.orientation.x = sr*cp*cy - cr*sp*sy
    pose.orientation.y = cr*sp*cy + sr*cp*sy
    pose.orientation.z = cr*cp*sy - sr*sp*cy
    pose.orientation.w = cr*cp*cy + sr*sp*sy
    published = []
    node = SimpleNamespace(frame='perception_floor', floor_z=0.,
                           broadcaster=SimpleNamespace(sendTransform=published.append))
    module.PerceptionFloorTf.on_odometry(node, odometry)
    transform = published[0]
    assert transform.header == odometry.header
    assert transform.child_frame_id == 'perception_floor'
    position = transform.transform.translation
    assert (position.x, position.y, position.z) == (3., -2., 0.)
    q = transform.transform.rotation
    assert (q.x, q.y) == (0., 0.)
    assert 2 * math.atan2(q.z, q.w) == pytest.approx(yaw)


def test_scan_callback_state_resets_and_empty_scan():
    node_module = load_node()
    from sensor_msgs.msg import LaserScan
    from types import SimpleNamespace

    class FakeDetector:
        def __init__(self, *args):
            self.resets = 0

        def reset(self):
            self.resets += 1

        def infer(self, scan, increment):
            assert np.isfinite(scan).all()
            return np.full((1, len(scan), 1), 3.), np.zeros((1, len(scan), 2))

    published = []
    logger = SimpleNamespace(error=lambda *a, **k: None, warning=lambda *a, **k: None)
    # Exercise actual callback and ROS messages without opening a DDS graph.
    node = SimpleNamespace(detector=FakeDetector(), grid=None, last_stamp=None,
                           publisher=SimpleNamespace(publish=published.append), get_logger=lambda: logger,
                           p=dict(expected_frame='base_link', state_timeout=0.5, stride=1,
                                  range_min=0.3, range_max=15., confidence_threshold=0.5,
                                  nms_distance=0.5, person_width=0.6, person_depth=0.6,
                                  person_height=1.7, ground_z=0., latency_budget_ms=15.))
    callback = lambda msg: node_module.DrSpaamCpu.on_scan(node, msg)
    scan = LaserScan(angle_min=-0.1, angle_increment=0.1, range_min=0.3, range_max=15.)
    scan.header.frame_id = 'base_link'
    scan.header.stamp.sec = 10
    scan.ranges = [2., float('inf'), 2.]
    callback(scan)
    assert published[-1].header == scan.header and published[-1].objects
    assert node.detector.resets == 1
    scan.header.stamp.nanosec = 100000000
    callback(scan)
    assert node.detector.resets == 1
    scan.header.stamp.sec = 9  # bag rewind clears recurrent memory
    callback(scan)
    assert node.detector.resets == 2
    scan.ranges = [float('nan'), float('inf'), 0.1]
    callback(scan)
    assert not published[-1].objects
    count = len(published)
    scan.header.frame_id = 'wrong_frame'
    callback(scan)
    assert len(published) == count
