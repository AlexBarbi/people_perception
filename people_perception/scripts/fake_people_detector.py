#!/usr/bin/env python3
"""
fake_people_detector.py (ROS 2 Python)

Stands in for autoware_lidar_centerpoint while there is no model (milestone M2 of docs/perception_pipeline.md), with the
same interface: for every cloud on input_cloud_topic it publishes one autoware_perception_msgs/DetectedObjects on
objects_topic with the header of that cloud (frame sensor_level, stamp of the scan), an empty one included.

It does not look at the points. The people come from a script file: it knows where they are supposed to be, not where
the lidar sees them. A script that does not match the scene makes the filter cut holes where nobody is.

Script file (yaml):

    frame: camera_init      # the waypoints are in this frame ...
    origin: [0.0, 0.0, 0.0] # ... or, with origin, in a frame whose origin is at x y [m] of `frame`, turned by yaw
                            # [rad]: waypoints copied from a Gazebo world, origin = where the world origin is
    start_time: 0.0         # stamp [s] at which the script time is 0
    people:
      - name: walker
        size: [0.5, 0.5, 1.7]         # box: length, width, height [m]
        floor_z: -0.47                # height of the floor under the person, in `frame` [m]
        loop: true                    # false: the person stays at the last waypoint
        delay_start: 0.0              # optional [s]: at the first waypoint until then, once (not before every lap)
        interpolation: linear         # optional. linear: straight lines at constant speed between the waypoints.
                                      # gazebo: as Gazebo moves a scripted <actor>: a Catmull-Rom spline through the
                                      # waypoints, slower near them, swinging past a waypoint that is repeated
        waypoints:                    # [t, x, y]
          - [0.0, 4.0, 2.0]
          - [6.0, 4.0, -2.0]

A person is detected when within max_range of the sensor, with probability detection_probability, at the scripted
position plus Gaussian noise of position_noise, as a PEDESTRIAN with existence_probability score. The answer is held
back by delay seconds, as an inference would. The clouds are answered in order; one whose TF (script frame to the
frame of the cloud, at its stamp) has not come waits for it up to tf_timeout, then gets an empty answer.

Every parameter is in config/fake_people_detector.yaml; script_file comes from the launch.
"""

import math
import time
from collections import deque

import numpy as np
import rclpy
import yaml
from autoware_perception_msgs.msg import DetectedObject, DetectedObjects, ObjectClassification, Shape
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener


class Person:
    def __init__(self, entry):
        self.name = entry["name"]
        self.size = [float(v) for v in entry["size"]]
        self.floor_z = float(entry["floor_z"])
        self.loop = bool(entry["loop"])
        self.delay_start = float(entry.get("delay_start", 0.0))
        self.interpolation = entry.get("interpolation", "linear")
        if self.interpolation not in ("linear", "gazebo"):
            raise ValueError(f"{self.name}: interpolation '{self.interpolation}' is not linear or gazebo")
        waypoints = np.array(entry["waypoints"], dtype=float)  # t x y
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or np.any(np.diff(waypoints[:, 0]) <= 0):
            raise ValueError(f"{self.name}: waypoints must be [t, x, y] rows with increasing t")
        self.times = waypoints[:, 0] - waypoints[0, 0]
        self.points = waypoints[:, 1:]
        # Tangents as gz-math Spline::RecalcTangents (tension 0) computes them for the key frames of the actor
        last = len(self.points) - 1
        self.tangents = np.zeros_like(self.points)
        self.tangents[1:last] = (self.points[2:] - self.points[:-2]) / 2.0
        if np.array_equal(self.points[0], self.points[last]) and last >= 2:  # closed: ends where it starts
            self.tangents[0] = self.tangents[last] = (self.points[1] - self.points[last - 1]) / 2.0
        else:
            self.tangents[0] = (self.points[1] - self.points[0]) / 2.0
            self.tangents[last] = (self.points[last] - self.points[last - 1]) / 2.0

    def position(self, t):
        """Position (x, y) at script time t."""
        length = self.times[-1]
        t = max(t - self.delay_start, 0.0)
        if self.loop and t > length:  # gz-sim ActorTrajectoryAt: while (time > totalTime) time -= totalTime
            t -= length * math.ceil(t / length - 1.0)
        t = min(t, length)
        i = min(max(int(np.searchsorted(self.times, t, side="right")) - 1, 0), len(self.times) - 2)
        u = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
        p0, p1 = self.points[i], self.points[i + 1]
        if self.interpolation == "linear":
            return p0 + (p1 - p0) * u
        return ((2 * u**3 - 3 * u**2 + 1) * p0 + (u**3 - 2 * u**2 + u) * self.tangents[i] +
                (-2 * u**3 + 3 * u**2) * p1 + (u**3 - u**2) * self.tangents[i + 1])

    def at(self, t):
        """Position (x, y) and heading at script time t. The heading is the way the person moves."""
        step = self.position(t + 0.05) - self.position(t - 0.05)
        return self.position(t), math.atan2(step[1], step[0])


class Script:
    def __init__(self, script):
        self.frame = script["frame"]
        self.start_time = float(script["start_time"])
        self.origin = [float(v) for v in script.get("origin", [0.0, 0.0, 0.0])]  # x y yaw of the script in frame
        self.people = [Person(entry) for entry in script.get("people") or []]

    def at(self, stamp):
        """(person, x, y, heading) of everybody at the stamp [s], in self.frame."""
        ox, oy, yaw = self.origin
        cos, sin = math.cos(yaw), math.sin(yaw)
        placed = []
        for person in self.people:
            (x, y), heading = person.at(stamp - self.start_time)
            placed.append((person, ox + cos * x - sin * y, oy + sin * x + cos * y, heading + yaw))
        return placed


def to_matrix(transform):
    t, q = transform.translation, transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = [t.x, t.y, t.z]
    return m


class FakePeopleDetector(Node):
    def __init__(self):
        super().__init__("fake_people_detector")

        def parameter(name, kind):  # no default: the yaml has every parameter
            value = self.declare_parameter(name, kind).value
            if value is None:
                raise RuntimeError(f"parameter {name} is not set")
            return value

        script_file = parameter("script_file", rclpy.Parameter.Type.STRING)
        self.tf_timeout = parameter("tf_timeout", rclpy.Parameter.Type.DOUBLE)
        self.max_range = parameter("max_range", rclpy.Parameter.Type.DOUBLE)
        self.detection_probability = parameter("detection_probability", rclpy.Parameter.Type.DOUBLE)
        self.position_noise = parameter("position_noise", rclpy.Parameter.Type.DOUBLE)
        self.score = parameter("score", rclpy.Parameter.Type.DOUBLE)
        self.delay = parameter("delay", rclpy.Parameter.Type.DOUBLE)
        self.random = np.random.default_rng(parameter("seed", rclpy.Parameter.Type.INTEGER))
        depth = parameter("qos_depth", rclpy.Parameter.Type.INTEGER)

        with open(script_file) as f:
            self.script = Script(yaml.safe_load(f))
        self.get_logger().info(f"{len(self.script.people)} scripted people in {self.script.frame} from {script_file}")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(DetectedObjects, parameter("objects_topic", rclpy.Parameter.Type.STRING),
                                               depth)
        self.create_subscription(PointCloud2, parameter("input_cloud_topic", rclpy.Parameter.Type.STRING),
                                 self.on_cloud, depth)
        self.waiting = deque()  # (header, wall time of arrival [ns]) of the clouds whose TF has not come, in order
        self.held = deque()     # (wall time to publish at [ns], message), in the order of the scans
        # Wall time, as an inference takes: it does not stop with a paused simulation
        self.create_timer(0.005, self.work, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_cloud(self, cloud: PointCloud2):
        self.waiting.append((cloud.header, time.monotonic_ns()))
        self.work()

    def work(self):
        """Answer, in order, the clouds whose TF is here; release the answers whose delay is over."""
        while self.waiting:
            header, arrived = self.waiting[0]
            msg = self.detect(header, give_up=time.monotonic_ns() - arrived > self.tf_timeout * 1e9)
            if msg is None:
                break  # its TF travels on its own and is not here yet: the later clouds wait behind it
            self.waiting.popleft()
            self.held.append((time.monotonic_ns() + int(self.delay * 1e9), msg))
        while self.held and self.held[0][0] <= time.monotonic_ns():
            self.publisher.publish(self.held.popleft()[1])

    def detect(self, header, give_up):
        """The people of the script as seen from the cloud with this header; None while its TF is missing."""
        msg = DetectedObjects()
        msg.header = header
        stamp = Time.from_msg(header.stamp)
        try:
            # No waiting here: the TF callback runs on the thread this would block
            sensor_from_script = to_matrix(
                self.tf_buffer.lookup_transform(header.frame_id, self.script.frame, stamp).transform)
        except TransformException as e:
            if not give_up:
                return None
            # The detector answers every cloud all the same: the filter waits for it
            self.get_logger().warn(f"no TF {header.frame_id} <- {self.script.frame} at the stamp of a cloud after "
                                   f"{self.tf_timeout} s, nobody detected: {e}", throttle_duration_sec=5.0)
            return msg
        frame_yaw = math.atan2(sensor_from_script[1, 0], sensor_from_script[0, 0])
        for person, x, y, heading in self.script.at(stamp.nanoseconds * 1e-9):
            centre = sensor_from_script @ np.array([x, y, person.floor_z + person.size[2] / 2.0, 1.0])
            if math.hypot(centre[0], centre[1]) > self.max_range or self.random.random() >= self.detection_probability:
                continue
            centre[:2] += self.random.normal(0.0, self.position_noise, 2) if self.position_noise > 0.0 else 0.0
            detected = DetectedObject()
            detected.existence_probability = self.score
            detected.classification = [ObjectClassification(label=ObjectClassification.PEDESTRIAN, probability=1.0)]
            pose = detected.kinematics.pose_with_covariance.pose
            pose.position.x, pose.position.y, pose.position.z = centre[:3].tolist()
            pose.orientation.z = math.sin((heading + frame_yaw) / 2.0)
            pose.orientation.w = math.cos((heading + frame_yaw) / 2.0)
            detected.shape.type = Shape.BOUNDING_BOX
            detected.shape.dimensions.x, detected.shape.dimensions.y, detected.shape.dimensions.z = person.size
            msg.objects.append(detected)
        return msg


def main():
    rclpy.init()
    node = FakePeopleDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
