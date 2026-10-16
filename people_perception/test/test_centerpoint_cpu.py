"""centerpoint_cpu.py: the numpy pre/post-processing against the CUDA kernels of autoware_lidar_centerpoint, written here
as plain loops (one thread = one loop iteration); the two networks only when the interpreter has onnxruntime and the
model folder exists (CENTERPOINT_MODEL, default ~/RaNAV/models/lidar_centerpoint/v4.1/short_range)."""
import importlib.util
import math
import os

import numpy as np
import pytest
from autoware_perception_msgs.msg import DetectedObjectKinematics, ObjectClassification
from geometry_msgs.msg import Transform
from sensor_msgs.msg import PointCloud2, PointField

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'centerpoint_cpu.py')
_spec = importlib.util.spec_from_file_location('centerpoint_cpu', _path)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

CLASSES = ['CAR', 'TRUCK', 'BUS', 'BICYCLE', 'PEDESTRIAN']
MODEL = os.environ.get('CENTERPOINT_MODEL', os.path.expanduser('~/RaNAV/models/lidar_centerpoint/v4.1/short_range'))


def config(half_range=5.12, encoder_features=10, max_voxels=64000, bins=(50.0, 90.0, 121.0, 200.0), threshold=0.35):
    return cp.Config(CLASSES, 4, max_voxels, [-half_range, -half_range, -3.0, half_range, half_range, 5.0],
                     [0.16, 0.16, 8.0], 4, encoder_features, list(bins),
                     {c: [threshold] * len(bins) for c in CLASSES}, 0.3, [0.3, 0.3, 0.3, 0.3, 0.0])


def reference_voxelize(points, cfg):
    """generateVoxels_random_kernel, generateBaseFeatures_kernel and generateFeatures_kernel, one point / cell at a time
    (points already in the order the shuffle left them)."""
    gx, gy = cfg.grid_size_x, cfg.grid_size_y
    mask = np.zeros(gx * gy, dtype=int)
    voxels = np.zeros((gx * gy, 32, 4), dtype=np.float32)
    for p in points:
        x, y, z = p[:3]
        # the kernel's comparisons let a NaN through (then an undefined cell); voxelize drops it, the clouds have none
        if not np.all(np.isfinite(p[:3])):
            continue
        if (x < cfg.range_min[0] or x >= cfg.range_max[0] or y < cfg.range_min[1] or y >= cfg.range_max[1]
                or z < cfg.range_min[2] or z >= cfg.range_max[2]):
            continue
        ix = min(max(int(math.floor((x - cfg.range_min[0]) / cfg.voxel_size[0])), 0), gx - 1)
        iy = min(max(int(math.floor((y - cfg.range_min[1]) / cfg.voxel_size[1])), 0), gy - 1)
        index = (gx - 1 - ix) * gy + iy
        if mask[index] < 32:
            voxels[index, mask[index]] = p
        mask[index] += 1
    features, coords = [], []
    for inverted in range(gx):                       # row-major over (inverted x, y): the pillar order
        for iy in range(gy):
            index = inverted * gy + iy
            count = min(mask[index], 32)
            if count == 0 or len(features) >= cfg.max_voxel_size:
                continue
            ix = gx - 1 - inverted
            pts = voxels[index, :count]
            mean = pts[:, :3].mean(axis=0)
            centre = (cfg.voxel_size[0] / 2 + ix * cfg.voxel_size[0] + cfg.range_min[0],
                      cfg.voxel_size[1] / 2 + iy * cfg.voxel_size[1] + cfg.range_min[1],
                      cfg.voxel_size[2] / 2 + cfg.range_min[2])
            f = np.zeros((32, cfg.encoder_in_feature_size), dtype=np.float32)
            for k in range(count):
                row = [*pts[k, :4], *(pts[k, :3] - mean), pts[k, 0] - centre[0], pts[k, 1] - centre[1]]
                if cfg.encoder_in_feature_size == 10:
                    row.append(pts[k, 2] - centre[2])
                f[k] = row
            features.append(f)
            coords.append((ix, iy))
    return np.array(features, dtype=np.float32).reshape(-1, 32, cfg.encoder_in_feature_size), coords


class NoShuffle:
    """An rng whose permutation keeps the order: the reference then sees the points as voxelize does."""
    def permutation(self, n):
        return np.arange(n)


def random_points(n, rng, half_range=6.0):
    pts = np.zeros((n, 4), dtype=np.float32)
    pts[:, 0:2] = rng.uniform(-half_range, half_range, (n, 2))
    pts[:, 2] = rng.uniform(-3.5, 5.5, n)
    pts[:, 3] = rng.choice([0.0, 0.1], n)
    # dense clusters: pillars with more than 32 points
    pts[: n // 4, 0:2] = rng.uniform(-0.3, 0.3, (n // 4, 2))
    return pts


@pytest.mark.parametrize('encoder_features', [9, 10])
def test_voxelize_matches_kernels(encoder_features):
    cfg = config(encoder_features=encoder_features)
    pts = random_points(3000, np.random.default_rng(1))
    pts[5] = [np.nan, 0.0, 0.0, 0.0]                  # NaN: never inside the range
    features, cx, cy, total = cp.voxelize(pts, cfg, NoShuffle())
    ref_features, ref_coords = reference_voxelize(pts, cfg)
    assert total == len(ref_coords)
    assert list(zip(cx.tolist(), cy.tolist())) == ref_coords
    np.testing.assert_allclose(features, ref_features, atol=1e-5)
    assert features.shape[1:] == (32, encoder_features)


def test_voxelize_max_voxel_size_keeps_the_first_pillars():
    cfg = config(max_voxels=50)
    pts = random_points(3000, np.random.default_rng(2))
    features, cx, cy, total = cp.voxelize(pts, cfg, NoShuffle())
    ref_features, ref_coords = reference_voxelize(pts, cfg)
    assert total > 50 and len(cx) == 50
    assert list(zip(cx.tolist(), cy.tolist())) == ref_coords
    np.testing.assert_allclose(features, ref_features, atol=1e-5)


def test_voxelize_keeps_32_random_points_of_a_full_pillar():
    cfg = config()
    pts = np.zeros((100, 4), dtype=np.float32)
    pts[:, 0] = 0.05
    pts[:, 1] = 0.05
    pts[:, 2] = np.linspace(-1, 1, 100)
    features, _, _, _ = cp.voxelize(pts, cfg, np.random.default_rng(0))
    z = features[0, :, 2]
    assert len(features) == 1 and len(np.unique(z)) == 32
    assert np.all(np.isin(z, pts[:, 2]))


def test_scatter_puts_each_pillar_in_its_cell():
    cfg = config()
    pf = np.arange(2 * 32, dtype=np.float32).reshape(2, 1, 32)
    spatial = cp.scatter(pf, np.array([3, 10]), np.array([7, 0]), cfg)
    assert spatial.shape == (1, 32, cfg.grid_size_y, cfg.grid_size_x)
    np.testing.assert_array_equal(spatial[0, :, 7, 3], pf[0, 0])
    np.testing.assert_array_equal(spatial[0, :, 0, 10], pf[1, 0])
    assert np.count_nonzero(spatial) == 63           # pillar 0 feature 0 is 0


def head_outputs(cfg, cells):
    """Head outputs with peaks at cells: (yi, xi, class, logit, offset x, offset y, z, w, l, h, yaw)."""
    h, w = cfg.grid_size_y // cfg.downsample_factor, cfg.grid_size_x // cfg.downsample_factor
    heatmap = np.full((1, cfg.class_size, h, w), -20.0, dtype=np.float32)
    offset, z = np.zeros((1, 2, h, w), np.float32), np.zeros((1, 1, h, w), np.float32)
    dim, rot = np.zeros((1, 3, h, w), np.float32), np.zeros((1, 2, h, w), np.float32)
    rot[0, 1] = 1.0
    for yi, xi, c, logit, ox, oy, zz, ww, ll, hh, yaw in cells:
        heatmap[0, c, yi, xi] = logit
        offset[0, :, yi, xi] = (ox, oy)
        z[0, 0, yi, xi] = zz
        dim[0, :, yi, xi] = np.log([ww, ll, hh])
        rot[0, :, yi, xi] = (math.sin(yaw), math.cos(yaw))
    return heatmap, offset, z, dim, rot


def test_decode_box_and_thresholds():
    cfg = config(bins=(3.0, 200.0))
    cfg.score_thresholds[1, 4] = 0.9                 # pedestrians beyond 3 m need a score of 0.9
    cells = [(9, 10, 4, 2.0, 0.25, 0.5, 0.3, 0.6, 0.7, 1.7, 0.4),   # score 0.88 at 1.7 m: kept
             (2, 2, 4, 2.0, 0.0, 0.0, 0.0, 0.6, 0.7, 1.7, 0.0),     # score 0.88 at 5.4 m: below 0.9
             (12, 12, 0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.5, 1.6, 0.0)]   # car, score 0.5 at 3.6 m: kept (0.35)
    boxes = cp.decode(*head_outputs(cfg, cells), cfg)
    assert len(boxes) == 2
    label, score, x, y, z, length, width, height, yaw = boxes[0]
    assert label == 4 and score == pytest.approx(1 / (1 + math.exp(-2.0)), rel=1e-5)
    assert x == pytest.approx(0.16 * 4 * (10 + 0.25) - 5.12, abs=1e-4)
    assert y == pytest.approx(0.16 * 4 * (9 + 0.5) - 5.12, abs=1e-4)
    assert (z, length, width, height) == pytest.approx((0.3, 0.7, 0.6, 1.7), rel=1e-5)
    assert yaw == pytest.approx(0.4, abs=1e-6)
    assert boxes[1][0] == 0                          # second by score


def test_decode_drops_beyond_the_last_distance_bin_and_low_yaw_norm():
    cfg = config(bins=(1.0,))
    cells = [(8, 8, 4, 3.0, 0.25, 0.5, 0.0, 0.6, 0.7, 1.7, 0.0),    # (0.16, 0.32) 0.36 m: inside the only bin
             (2, 2, 4, 3.0, 0.0, 0.0, 0.0, 0.6, 0.7, 1.7, 0.0)]     # 5.4 m: beyond it
    heatmap, offset, z, dim, rot = head_outputs(cfg, cells)
    assert len(cp.decode(heatmap, offset, z, dim, rot, cfg)) == 1
    rot[0, :, 8, 8] = (0.1, 0.1)                     # yaw norm 0.14: below the car threshold 0.3, a pedestrian's is 0
    assert len(cp.decode(heatmap, offset, z, dim, rot, cfg)) == 1
    heatmap[0, 4, 8, 8], heatmap[0, 0, 8, 8] = -20, 3.0   # the same cell as a car
    assert len(cp.decode(heatmap, offset, z, dim, rot, cfg)) == 0


def test_circle_nms_keeps_the_best_of_close_boxes():
    boxes = np.array([[4, 0.9, 0.0, 0.0, 0, 1, 1, 1, 0], [4, 0.8, 0.2, 0.0, 0, 1, 1, 1, 0],
                      [4, 0.7, 0.45, 0.0, 0, 1, 1, 1, 0], [0, 0.6, 5.0, 5.0, 0, 1, 1, 1, 0]])
    kept = cp.circle_nms(boxes, 0.3)
    # 0.2 m from the first: removed; 0.45 m from the first, 0.25 m from a removed one: kept
    assert kept[:, 1].tolist() == [0.9, 0.7, 0.6]


def objects(*boxes):
    return [cp.box_to_object(np.array(b, dtype=np.float64), CLASSES) for b in boxes]


def test_iou_bev_nms():
    cars = objects([0, 0.9, 0.0, 0.0, 0, 4.0, 2.0, 1.5, 0.0], [0, 0.8, 0.5, 0.0, 0, 4.0, 2.0, 1.5, 0.1],
                   [0, 0.7, 30.0, 0.0, 0, 4.0, 2.0, 1.5, 0.0])
    kept = cp.iou_bev_nms(cars, 10.0, 0.1)
    assert [o.existence_probability for o in kept] == pytest.approx([0.9, 0.7])
    mixed = objects([0, 0.9, 0.0, 0.0, 0, 4.0, 2.0, 1.5, 0.0], [4, 0.8, 0.2, 0.0, 0, 0.7, 0.6, 1.7, 0.0])
    assert len(cp.iou_bev_nms(mixed, 10.0, 0.1)) == 2   # a pedestrian is compared with pedestrians only


def test_bev_iou_of_rotated_boxes():
    a = cp.box_corners(0, 0, 0.0, 2.0, 2.0)
    assert cp.bev_iou(a, cp.box_corners(0, 0, 0.0, 2.0, 2.0)) == pytest.approx(1.0)
    assert cp.bev_iou(a, cp.box_corners(1, 0, 0.0, 2.0, 2.0)) == pytest.approx(1 / 3)
    # square turned 45 deg on the same centre: intersection is the octagon 8 (sqrt(2) - 1) of area
    assert cp.bev_iou(a, cp.box_corners(0, 0, math.pi / 4, 2.0, 2.0)) == pytest.approx(
        8 * (math.sqrt(2) - 1) / (8 - 8 * (math.sqrt(2) - 1)), rel=1e-6)
    assert cp.bev_iou(a, cp.box_corners(5, 5, 0.3, 2.0, 2.0)) == 0.0


def test_box_to_object_conventions():
    car, ped = objects([0, 0.9, 1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.3], [4, 0.8, 0.0, 0.0, 0.0, 0.7, 0.6, 1.7, 0.3])
    pose = car.kinematics.pose_with_covariance.pose
    assert 2 * math.atan2(pose.orientation.z, pose.orientation.w) == pytest.approx(-0.3 - math.pi / 2)
    assert (pose.position.x, pose.position.y, pose.position.z) == (1.0, 2.0, 0.5)
    assert (car.shape.dimensions.x, car.shape.dimensions.y) == (4.0, 2.0)
    assert car.classification[0].label == ObjectClassification.CAR
    assert car.kinematics.orientation_availability == DetectedObjectKinematics.SIGN_UNKNOWN
    assert ped.classification[0].label == ObjectClassification.PEDESTRIAN
    assert ped.kinematics.orientation_availability == DetectedObjectKinematics.UNAVAILABLE


def test_class_remapper():
    n = 8
    allow, min_area, max_area = [0] * n * n, [0.0] * n * n, [0.0] * n * n
    car, truck = ObjectClassification.CAR, ObjectClassification.TRUCK
    allow[car * n + truck], min_area[car * n + truck], max_area[car * n + truck] = 1, 12.0, float('inf')
    remapper = cp.ClassRemapper(allow, min_area, max_area)
    small, big = objects([0, 0.9, 0, 0, 0, 4.0, 2.0, 1.5, 0], [0, 0.9, 0, 0, 0, 8.0, 2.5, 3.0, 0])
    remapper.apply([small, big])
    assert small.classification[0].label == car and big.classification[0].label == truck


@pytest.mark.skipif(not os.path.isfile(os.path.join(MODEL, 'detection_class_remapper.param.yaml')),
                    reason='no model folder')
def test_class_remapper_reads_the_model_file():
    remapper = cp.ClassRemapper.from_yaml(os.path.join(MODEL, 'detection_class_remapper.param.yaml'))
    assert remapper.allow.shape == (8, 8)


def xyzirc_cloud(xyz, fields=None):
    msg = PointCloud2()
    msg.height, msg.width, msg.point_step = 1, len(xyz), 16
    msg.fields = fields or [PointField(name=n, offset=o, datatype=d, count=1) for n, o, d in cp.XYZIRC_FIELDS]
    data = np.zeros(len(xyz), dtype=np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                                              'offsets': [0, 4, 8], 'itemsize': 16}))
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    msg.data = data.tobytes()
    msg.row_step = msg.point_step * msg.width
    return msg


def test_cloud_xyz():
    xyz = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    np.testing.assert_array_equal(cp.cloud_xyz(xyzirc_cloud(xyz)), xyz)
    xyzi = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
            for n, o in (('x', 0), ('y', 4), ('z', 8), ('intensity', 12))]
    assert cp.cloud_xyz(xyzirc_cloud(xyz, xyzi)) is None


def test_sweep_points_brings_past_scans_into_the_current_frame():
    t = Transform()
    t.translation.x = 1.0
    t.rotation.z, t.rotation.w = math.sin(math.pi / 4), math.cos(math.pi / 4)   # 90 deg about z
    world_to_past = cp.transform_matrix(t)
    world_to_current = np.eye(4)
    past = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    current = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    points = cp.sweep_points([(current, 10.1, world_to_current), (past, 10.0, world_to_past)], 100)
    # past point (1, 0, 0) in the past frame: world = R^T (p - t) = R^T (0, 0, 0) = origin
    np.testing.assert_allclose(points, [[0, 0, 1, 0], [0, 0, 0, 0.1]], atol=1e-6)


def test_config_rejects_a_grid_the_backbone_cannot_take():
    with pytest.raises(ValueError):
        config(half_range=6.0)                       # 75 cells


@pytest.mark.skipif(importlib.util.find_spec('onnxruntime') is None or not os.path.isdir(MODEL),
                    reason='needs onnxruntime and the model folder')
def test_networks_run_on_a_cloud():
    cfg = config(half_range=12.8)
    detector = cp.CenterPointCpu(os.path.join(MODEL, 'pts_voxel_encoder.onnx'),
                                 os.path.join(MODEL, 'pts_backbone_neck_head.onnx'), cfg, num_threads=4, seed=0)
    rng = np.random.default_rng(3)
    floor = np.column_stack([rng.uniform(-12, 12, (20000, 2)), np.full(20000, -0.47), np.zeros(20000)])
    boxes, num_pillars, times = detector.detect(floor.astype(np.float32))
    assert num_pillars > 1000 and boxes.shape[1] == 9 and len(times) == 3
