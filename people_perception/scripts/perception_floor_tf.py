#!/usr/bin/env python3
"""Publish a yaw-aligned floor frame from robot odometry on a known flat floor."""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class PerceptionFloorTf(Node):
    def __init__(self):
        super().__init__('perception_floor_tf')
        self.frame = self.declare_parameter('floor_frame', 'perception_floor').value
        self.floor_z = self.declare_parameter('floor_z', 0.0).value
        topic = self.declare_parameter('odometry_topic', '/odom').value
        self.broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(Odometry, topic, self.on_odometry, 10)

    def on_odometry(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        transform = TransformStamped()
        transform.header = msg.header
        transform.child_frame_id = self.frame
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = self.floor_z
        transform.transform.rotation.z = math.sin(yaw / 2)
        transform.transform.rotation.w = math.cos(yaw / 2)
        self.broadcaster.sendTransform(transform)


def main():
    rclpy.init()
    node = PerceptionFloorTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
