#!/usr/bin/env python3
"""
export_detector_dataset.py

Turns a bag into the raw data of a mmdetection3d custom dataset, to train the CenterPoint model of
docs/perception_pipeline.md (section 5) on exactly what its node will receive: the clouds of
/perception/detector/pointcloud (frame sensor_level, FAST-LIO decimation and the crop of detector_cloud_adapter).
Record that topic and /tf; to make it from the raw driver topics instead, play the bag through FAST-LIO and
detector_cloud_adapter while recording.

    <out_dir>/points/000000.bin   float32, N x 4: x y z time_lag. time_lag is 0: one scan per frame
    <out_dir>/labels/000000.txt   one box per line: x y z dx dy dz yaw Pedestrian, in the frame of the cloud;
                                  x y z is the centre of the bottom face, as mmdetection3d reads a custom label
                                  (LiDARInstance3DBoxes), dx dy dz are length, width, height
    <out_dir>/ImageSets/train.txt, val.txt   frame names; validation is the last val_fraction of the bag
    <out_dir>/frames.csv          per frame: stamp of the scan, pose of the cloud frame in the odometry frame
                                  (x y z yaw, from the TF detector_cloud_adapter publishes with every cloud),
                                  number of points and of labels

The labels come from --objects-topic when the bag has it: the answers of a detector to those same clouds. With the
fake detector on the Gazebo actors they are the truth of the simulation; with a real model they are a first guess to
correct by hand. Without that topic there is no labels/ folder: the frames are to be annotated, in the format above.

A scan is exported when --every-seconds have passed or the sensor has moved --every-meters since the last exported
one, whichever comes first; with neither option, every scan. A scan whose TF is not in the bag is left out.

Then, in the mmdetection3d fork of Autoware (github.com/autowarefoundation/mmdetection3d):
    python tools/create_data.py custom --root-path <out_dir> --out-dir <out_dir> --extra-tag custom
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointField

PEDESTRIAN = 7  # autoware_perception_msgs/ObjectClassification


def stamp_ns(header):
    return header.stamp.sec * 10**9 + header.stamp.nanosec


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def read(bag, topics):
    """(topic, message) of the given topics, in the order of the bag."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=''), rosbag2_py.ConverterOptions('', ''))
    types = {info.name: info.type for info in reader.get_all_topics_and_types()}
    wanted = [topic for topic in topics if topic in types]
    if wanted:
        reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))
        while reader.has_next():
            topic, data, _ = reader.read_next()
            yield topic, deserialize_message(data, get_message(types[topic]))


def topics_of(bag):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=''), rosbag2_py.ConverterOptions('', ''))
    return {info.name for info in reader.get_all_topics_and_types()}


def xyz_of(cloud):
    """The x y z of a cloud, (N, 3) float32."""
    offsets = {f.name: f.offset for f in cloud.fields if f.datatype == PointField.FLOAT32 and f.count == 1}
    if not all(name in offsets for name in 'xyz'):
        raise ValueError('cloud without FLOAT32 x y z fields')
    count = cloud.height * cloud.width
    points = np.frombuffer(cloud.data, dtype=np.uint8, count=count * cloud.point_step).reshape(count, cloud.point_step)
    return np.column_stack([points[:, offsets[name]:offsets[name] + 4].copy().view('<f4').ravel() for name in 'xyz'])


def export(bag, out_dir, every_seconds=None, every_meters=None, cloud_topic='/perception/detector/pointcloud',
           objects_topic='/perception/detector/objects', odom_frame='camera_init', class_name='Pedestrian',
           val_fraction=0.2):
    """Write the dataset of `bag` to `out_dir`. Returns the rows of frames.csv."""
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f'{out_dir} is not empty: a dataset is never written over another one')
    has_objects = objects_topic in topics_of(bag)

    poses, boxes = {}, {}  # by (frame of the cloud, stamp [ns]) and by stamp [ns]
    for topic, msg in read(bag, ['/tf', objects_topic]):
        if topic == '/tf':
            for transform in msg.transforms:
                if transform.header.frame_id == odom_frame:
                    t = transform.transform.translation
                    poses[(transform.child_frame_id, stamp_ns(transform.header))] = (
                        t.x, t.y, t.z, yaw_of(transform.transform.rotation))
        else:
            lines = []
            for detected in msg.objects:
                best = max(detected.classification, key=lambda c: c.probability, default=None)
                if best is None or best.label != PEDESTRIAN:
                    continue
                pose, size = detected.kinematics.pose_with_covariance.pose, detected.shape.dimensions
                lines.append(f'{pose.position.x:.4f} {pose.position.y:.4f} {pose.position.z - size.z / 2.0:.4f} '
                             f'{size.x:.4f} {size.y:.4f} {size.z:.4f} {yaw_of(pose.orientation):.4f} {class_name}')
            boxes[stamp_ns(msg.header)] = lines

    os.makedirs(os.path.join(out_dir, 'points'))
    os.makedirs(os.path.join(out_dir, 'ImageSets'))
    if has_objects:
        os.makedirs(os.path.join(out_dir, 'labels'))
    frames, without_transform, last = [], 0, None
    for _, cloud in read(bag, [cloud_topic]):
        stamp = stamp_ns(cloud.header)
        pose = poses.get((cloud.header.frame_id, stamp))
        if pose is None:
            without_transform += 1
            continue
        if last is not None and (every_seconds is not None or every_meters is not None):
            late = every_seconds is not None and stamp - last[0] >= every_seconds * 1e9
            far = every_meters is not None and math.hypot(pose[0] - last[1][0], pose[1] - last[1][1]) >= every_meters
            if not (late or far):
                continue
        last = (stamp, pose)
        name = f'{len(frames):06d}'
        xyz = xyz_of(cloud)
        np.column_stack([xyz, np.zeros(len(xyz), dtype=np.float32)]).astype('<f4').tofile(
            os.path.join(out_dir, 'points', name + '.bin'))
        labels = ''
        if has_objects:
            lines = boxes.get(stamp, [])
            with open(os.path.join(out_dir, 'labels', name + '.txt'), 'w') as f:
                f.write(''.join(line + '\n' for line in lines))
            labels = len(lines)
        frames.append({'frame': name, 'stamp_sec': stamp // 10**9, 'stamp_nanosec': stamp % 10**9,
                       'frame_id': cloud.header.frame_id, 'odom_frame': odom_frame, 'x': pose[0], 'y': pose[1],
                       'z': pose[2], 'yaw': pose[3], 'points': len(xyz), 'labels': labels})

    with open(os.path.join(out_dir, 'frames.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'stamp_sec', 'stamp_nanosec', 'frame_id', 'odom_frame', 'x', 'y',
                                               'z', 'yaw', 'points', 'labels'])
        writer.writeheader()
        writer.writerows(frames)
    # Validation is the end of the bag, not scattered frames: neighbouring scans are nearly the same scene
    validation = min(len(frames), max(1, round(len(frames) * val_fraction))) if val_fraction > 0 and len(frames) > 1 else 0
    names = [frame['frame'] for frame in frames]
    for split, part in (('train', names[:len(names) - validation]), ('val', names[len(names) - validation:])):
        with open(os.path.join(out_dir, 'ImageSets', split + '.txt'), 'w') as f:
            f.write(''.join(name + '\n' for name in part))
    if without_transform:
        print(f'{without_transform} scans left out: no TF {odom_frame} -> their frame at their stamp in the bag',
              file=sys.stderr)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag')
    parser.add_argument('out_dir')
    parser.add_argument('--every-seconds', type=float, help='export a scan this long after the last exported one')
    parser.add_argument('--every-meters', type=float, help='export a scan this far from the last exported one')
    parser.add_argument('--cloud-topic', default='/perception/detector/pointcloud')
    parser.add_argument('--objects-topic', default='/perception/detector/objects')
    parser.add_argument('--odom-frame', default='camera_init')
    parser.add_argument('--class-name', default='Pedestrian')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    args = parser.parse_args()
    frames = export(args.bag, args.out_dir, args.every_seconds, args.every_meters, args.cloud_topic,
                    args.objects_topic, args.odom_frame, args.class_name, args.val_fraction)
    labelled = [frame['labels'] for frame in frames if frame['labels'] != '']
    print(f'{len(frames)} frames in {args.out_dir}'
          + (f', {sum(labelled)} boxes, {sum(1 for n in labelled if n == 0)} frames without people' if labelled
             else ', no labels: to be annotated'))


if __name__ == '__main__':
    main()
