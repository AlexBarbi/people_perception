"""export_detector_dataset.py on small bags written here: what the detector node receives, as a mmdetection3d custom
dataset (points/*.bin, labels/*.txt, ImageSets) with the pose of every frame."""
import csv
import importlib.util
import math
import os

import numpy as np
import pytest
import rosbag2_py
from autoware_perception_msgs.msg import DetectedObject, DetectedObjects, ObjectClassification, Shape
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import serialize_message
from sensor_msgs.msg import PointCloud2, PointField
from tf2_msgs.msg import TFMessage

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'export_detector_dataset.py')
_spec = importlib.util.spec_from_file_location('export_detector_dataset', _path)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

CLOUD, OBJECTS = '/perception/detector/pointcloud', '/perception/detector/objects'


def stamp_of(msg, ns):
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 10**9)


def detector_cloud(ns, xyz):
    """A PointXYZIRC cloud in sensor_level, as detector_cloud_adapter publishes it."""
    msg = PointCloud2()
    msg.header.frame_id = 'sensor_level'
    stamp_of(msg, ns)
    msg.height, msg.width = 1, len(xyz)
    msg.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                  PointField(name='intensity', offset=12, datatype=PointField.UINT8, count=1),
                  PointField(name='return_type', offset=13, datatype=PointField.UINT8, count=1),
                  PointField(name='channel', offset=14, datatype=PointField.UINT16, count=1)]
    msg.point_step, msg.row_step, msg.is_dense = 16, 16 * len(xyz), True
    data = np.zeros(len(xyz), dtype=np.dtype({'names': ['x', 'y', 'z', 'i'], 'formats': ['<f4', '<f4', '<f4', 'u1'],
                                              'offsets': [0, 4, 8, 12], 'itemsize': 16}))
    data['x'], data['y'], data['z'] = np.asarray(xyz, dtype=np.float32).reshape(-1, 3).T
    data['i'] = 200
    msg.data = data.tobytes()
    return msg


def level_transform(ns, x, y, z, yaw):
    transform = TransformStamped()
    transform.header.frame_id, transform.child_frame_id = 'camera_init', 'sensor_level'
    stamp_of(transform, ns)
    transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = x, y, z
    transform.transform.rotation.z, transform.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return TFMessage(transforms=[transform])


def detections(ns, boxes):
    msg = DetectedObjects()
    msg.header.frame_id = 'sensor_level'
    stamp_of(msg, ns)
    for x, y, z, length, width, height, yaw in boxes:
        detected = DetectedObject()
        detected.classification = [ObjectClassification(label=ObjectClassification.PEDESTRIAN, probability=1.0)]
        pose = detected.kinematics.pose_with_covariance.pose
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation.z, pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        detected.shape.type = Shape.BOUNDING_BOX
        detected.shape.dimensions.x, detected.shape.dimensions.y, detected.shape.dimensions.z = length, width, height
        msg.objects.append(detected)
    return msg


def write_bag(path, scans, with_objects=True):
    """scans: dicts with ns, xyz, and optionally pose (x, y, z, yaw; None: no transform) and boxes."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    topics = {CLOUD: 'sensor_msgs/msg/PointCloud2', '/tf': 'tf2_msgs/msg/TFMessage',
              OBJECTS: 'autoware_perception_msgs/msg/DetectedObjects'}
    for i, (name, kind) in enumerate(topics.items()):
        if name != OBJECTS or with_objects:
            writer.create_topic(rosbag2_py.TopicMetadata(id=i, name=name, type=kind, serialization_format='cdr'))
    for scan in scans:
        ns = scan['ns']
        pose = scan.get('pose', (0.0, 0.0, 0.0, 0.0))
        if pose is not None:
            writer.write('/tf', serialize_message(level_transform(ns, *pose)), ns)
        writer.write(CLOUD, serialize_message(detector_cloud(ns, scan['xyz'])), ns + 1000)
        if with_objects:
            writer.write(OBJECTS, serialize_message(detections(ns, scan.get('boxes', []))), ns + 2000)
    del writer
    return str(path)


def rows_of(out):
    with open(os.path.join(out, 'frames.csv')) as f:
        return list(csv.DictReader(f))


def lines_of(path):
    with open(path) as f:
        return [line for line in f.read().splitlines() if line]


def test_every_scan_becomes_points_a_pose_and_an_index_row(tmp_path):
    xyz = [[1.0, 2.0, 0.5], [-3.0, 0.25, 1.5]]
    bag = write_bag(tmp_path / 'bag', [
        {'ns': 10_700_000_000, 'xyz': xyz, 'pose': (4.0, -1.0, 0.47, 0.3)},
        {'ns': 10_800_000_000, 'xyz': xyz[:1], 'pose': (4.1, -1.0, 0.47, 0.3)},
        {'ns': 10_900_000_000, 'xyz': [], 'pose': (4.2, -1.0, 0.47, 0.3)}])
    out = str(tmp_path / 'dataset')
    frames = tool.export(bag, out)

    assert len(frames) == 3
    points = np.fromfile(os.path.join(out, 'points', '000000.bin'), dtype=np.float32).reshape(-1, 4)
    assert points[:, :3].tolist() == xyz
    assert points[:, 3].tolist() == [0.0, 0.0]  # time lag: a single scan
    assert np.fromfile(os.path.join(out, 'points', '000001.bin'), dtype=np.float32).size == 4
    assert os.path.getsize(os.path.join(out, 'points', '000002.bin')) == 0  # an empty scan is a frame too

    rows = rows_of(out)
    assert [row['frame'] for row in rows] == ['000000', '000001', '000002']
    assert (rows[0]['stamp_sec'], rows[0]['stamp_nanosec']) == ('10', '700000000')
    assert [float(rows[0][k]) for k in ('x', 'y', 'z', 'yaw')] == pytest.approx([4.0, -1.0, 0.47, 0.3])
    assert rows[0]['frame_id'] == 'sensor_level' and rows[0]['odom_frame'] == 'camera_init'
    assert int(rows[0]['points']) == 2

    train = lines_of(os.path.join(out, 'ImageSets', 'train.txt'))
    val = lines_of(os.path.join(out, 'ImageSets', 'val.txt'))
    assert train + val == ['000000', '000001', '000002']  # in time, validation last: no leak between neighbours
    assert val


def test_one_scan_every_half_second(tmp_path):
    bag = write_bag(tmp_path / 'bag', [{'ns': 20_000_000_000 + i * 100_000_000, 'xyz': [[1.0, 0.0, 0.0]]}
                                       for i in range(12)])
    out = str(tmp_path / 'dataset')
    tool.export(bag, out, every_seconds=0.5)
    assert [(row['stamp_sec'], row['stamp_nanosec']) for row in rows_of(out)] == [
        ('20', '0'), ('20', '500000000'), ('21', '0')]


def test_one_scan_every_half_meter(tmp_path):
    bag = write_bag(tmp_path / 'bag', [{'ns': 20_000_000_000 + i * 100_000_000, 'xyz': [[1.0, 0.0, 0.0]],
                                        'pose': (0.2 * i, 0.0, 0.47, 0.0)} for i in range(10)])
    out = str(tmp_path / 'dataset')
    tool.export(bag, out, every_meters=0.5)
    assert [float(row['x']) for row in rows_of(out)] == pytest.approx([0.0, 0.6, 1.2, 1.8])


def test_either_half_a_second_or_half_a_meter(tmp_path):
    # Standing for a second, then 1 m in 0.1 s: exported by time while standing, by distance on the jump
    scans = [{'ns': 20_000_000_000 + i * 100_000_000, 'xyz': [[1.0, 0.0, 0.0]], 'pose': (0.0, 0.0, 0.47, 0.0)}
             for i in range(8)]
    scans.append({'ns': 20_800_000_000, 'xyz': [[1.0, 0.0, 0.0]], 'pose': (1.0, 0.0, 0.47, 0.0)})
    out = str(tmp_path / 'dataset')
    tool.export(write_bag(tmp_path / 'bag', scans), out, every_seconds=0.5, every_meters=0.5)
    assert [(row['stamp_nanosec'], float(row['x'])) for row in rows_of(out)] == [
        ('0', 0.0), ('500000000', 0.0), ('800000000', 1.0)]


# mmdetection3d reads a custom label as a LiDAR box, whose x y z is the centre of its bottom face.
def test_boxes_of_the_bag_become_bottom_centre_labels(tmp_path):
    bag = write_bag(tmp_path / 'bag', [
        {'ns': 10_700_000_000, 'xyz': [[2.0, 1.0, 0.4]], 'boxes': [(2.0, 1.0, 0.43, 0.6, 0.5, 1.8, 0.3),
                                                                     (-4.0, 2.5, 0.38, 0.6, 0.6, 1.7, -1.2)]},
        {'ns': 10_800_000_000, 'xyz': [[2.0, 1.0, 0.4]], 'boxes': []}])
    out = str(tmp_path / 'dataset')
    tool.export(bag, out)

    labels = [line.split() for line in lines_of(os.path.join(out, 'labels', '000000.txt'))]
    assert [float(v) for v in labels[0][:7]] == pytest.approx([2.0, 1.0, 0.43 - 0.9, 0.6, 0.5, 1.8, 0.3], abs=1e-4)
    assert labels[0][7] == 'Pedestrian'
    assert [float(v) for v in labels[1][:7]] == pytest.approx([-4.0, 2.5, 0.38 - 0.85, 0.6, 0.6, 1.7, -1.2], abs=1e-4)
    assert lines_of(os.path.join(out, 'labels', '000001.txt')) == []  # a scene without people is a negative
    assert [int(row['labels']) for row in rows_of(out)] == [2, 0]


def test_bag_without_detections_gives_no_label_files(tmp_path):
    bag = write_bag(tmp_path / 'bag', [{'ns': 10_700_000_000, 'xyz': [[2.0, 1.0, 0.4]]}], with_objects=False)
    out = str(tmp_path / 'dataset')
    tool.export(bag, out)
    assert not os.path.exists(os.path.join(out, 'labels'))  # to be annotated by hand
    assert rows_of(out)[0]['labels'] == ''


def test_scan_without_its_transform_is_left_out(tmp_path):
    bag = write_bag(tmp_path / 'bag', [
        {'ns': 10_700_000_000, 'xyz': [[2.0, 1.0, 0.4]]},
        {'ns': 10_800_000_000, 'xyz': [[2.0, 1.0, 0.4]], 'pose': None},
        {'ns': 10_900_000_000, 'xyz': [[2.0, 1.0, 0.4]]}])
    out = str(tmp_path / 'dataset')
    frames = tool.export(bag, out)
    assert [row['stamp_nanosec'] for row in rows_of(out)] == ['700000000', '900000000']
    assert len(frames) == 2


def test_refuses_to_write_into_a_dataset_that_is_there(tmp_path):
    bag = write_bag(tmp_path / 'bag', [{'ns': 10_700_000_000, 'xyz': [[2.0, 1.0, 0.4]]}])
    out = str(tmp_path / 'dataset')
    tool.export(bag, out)
    with pytest.raises(FileExistsError):
        tool.export(bag, out)
