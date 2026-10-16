#!/usr/bin/env python3
"""
centerpoint_cpu.py (ROS 2 Python)

autoware_lidar_centerpoint without CUDA and TensorRT: the same CenterPoint model (the two ONNX files of
centerpoint_model), the same parameter files and the same topics, run with ONNX Runtime on the CPU. For a computer
without an NVIDIA GPU (the NUC of the robot). people_perception_launch.xml starts it in place of the TensorRT node with
detector:=centerpoint_cpu.

Every step is the one of autoware_lidar_centerpoint (commit 8413d41), CUDA kernels rewritten with numpy:
  densification     the previous scan(s), brought into the current frame through densification_params.world_frame_id,
                    their age as 4th feature                                     (pointcloud_densification.cpp)
  pillars           random order, at most 32 points per 0.16 x 0.16 m pillar     (generateVoxels_random_kernel)
  pillar features   point, offset to the pillar mean, offset to the pillar centre (generateFeatures_kernel)
  voxel encoder     pts_voxel_encoder.onnx
  scatter           pillar features into the bird's-eye grid                     (scatterFeatures_kernel)
  backbone + head   pts_backbone_neck_head.onnx
  decode            heatmap peaks above the class and distance threshold        (generateBoxes3D_kernel)
  circle NMS        boxes closer than circle_nms_dist_threshold                  (circleNMS_Kernel)
  IoU BEV NMS       perception_utils::IouBevNms, then the class remapper of the model folder

The one difference is the area: cpu.point_cloud_range (centerpoint_cpu.param.yaml) replaces the model's
model_params.point_cloud_range. The time follows the grid: on a Core Ultra 7 265H, 1.6 s per scan at the model's
+-51.2 m (640 x 640), 84 ms at +-12.8 m (160 x 160, 4 threads). The network is fully convolutional: no retraining, the
decoder takes the origin of the smaller grid. Each side of the grid must be a multiple of 32 cells (5.12 m). On 300
simulated scans the pedestrians agree with the TensorRT node's as much at +-12.8 m as at the full range (80 %); the
others are borderline scores (FP32 here, FP16 there).

Scans arriving while one is processed are cached for the densification and only the newest is processed next.

Needs onnxruntime (pip): run with the python of a venv that has it, e.g. the .venv-yolo of the workspace (the launch
does, launch-prefix). Usage: ros2 launch fast_lio fast_lio_slam_launch.xml people:=true detector:=centerpoint_cpu
"""

import math
import threading
import time
from collections import deque

import numpy as np
import rclpy
import yaml
from autoware_perception_msgs.msg import DetectedObject, DetectedObjectKinematics, DetectedObjects
from autoware_perception_msgs.msg import ObjectClassification, Shape
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

MAX_POINT_IN_VOXEL = 32          # max_point_in_voxel_size_ of CenterPointConfig, fixed in the kernels too
LABEL_OF_CLASS = {
    "CAR": ObjectClassification.CAR, "TRUCK": ObjectClassification.TRUCK, "BUS": ObjectClassification.BUS,
    "TRAILER": ObjectClassification.TRAILER, "BICYCLE": ObjectClassification.BICYCLE,
    "MOTORBIKE": ObjectClassification.MOTORCYCLE, "PEDESTRIAN": ObjectClassification.PEDESTRIAN,
}
CAR_LIKE = {ObjectClassification.CAR, ObjectClassification.TRUCK, ObjectClassification.BUS,
            ObjectClassification.TRAILER}


class Config:
    """CenterPointConfig: the model geometry and thresholds, from the parameters of the TensorRT node."""

    def __init__(self, class_names, point_feature_size, max_voxel_size, point_cloud_range, voxel_size,
                 downsample_factor, encoder_in_feature_size, distance_bin_upper_limits, score_thresholds_by_class,
                 circle_nms_dist_threshold, yaw_norm_thresholds, cloud_capacity=2000000):
        if point_feature_size != 4 or encoder_in_feature_size not in (9, 10):
            raise ValueError("only point_feature_size 4 (x, y, z, time lag) with 9 or 10 encoder features is supported")
        self.class_names = list(class_names)
        self.class_size = len(self.class_names)
        self.encoder_in_feature_size = encoder_in_feature_size
        self.max_voxel_size = max_voxel_size
        self.cloud_capacity = cloud_capacity
        self.range_min = np.array(point_cloud_range[:3], dtype=np.float32)
        self.range_max = np.array(point_cloud_range[3:], dtype=np.float32)
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        # float32 and truncated, as CenterPointConfig computes it
        self.grid_size_x = int((self.range_max[0] - self.range_min[0]) / self.voxel_size[0])
        self.grid_size_y = int((self.range_max[1] - self.range_min[1]) / self.voxel_size[1])
        self.downsample_factor = downsample_factor
        if self.grid_size_x % 32 or self.grid_size_y % 32:
            raise ValueError(f"grid {self.grid_size_x} x {self.grid_size_y}: each side must be a multiple of 32 cells "
                             f"({32 * self.voxel_size[0]:.2f} m) for the backbone")
        self.distance_bin_upper_limits = np.array(distance_bin_upper_limits, dtype=np.float32)
        if np.any(np.diff(self.distance_bin_upper_limits) < 0):
            raise ValueError("distance_bin_upper_limits must be sorted in ascending order")
        # [distance bin, class], a threshold out of [0, 1) is 0 (as CenterPointConfig)
        thresholds = np.array([score_thresholds_by_class[name] for name in self.class_names], dtype=np.float32).T
        if thresholds.shape != (len(self.distance_bin_upper_limits), self.class_size):
            raise ValueError("each class needs one score threshold per distance bin")
        self.score_thresholds = np.where((thresholds >= 0) & (thresholds < 1), thresholds, 0).astype(np.float32)
        yaw = np.array(yaw_norm_thresholds, dtype=np.float32)
        self.yaw_norm_thresholds = np.where((yaw >= 0) & (yaw < 1), yaw, 0).astype(np.float32)
        self.circle_nms_dist_threshold = circle_nms_dist_threshold if circle_nms_dist_threshold > 0 else 1.5


# ------------------------------------------------------------------------------------------------------- pre-process
def voxelize(points, cfg, rng):
    """Pillars of points (N x 4: x, y, z, time lag) as generateVoxels_random + generateBaseFeatures + generateFeatures:
    the encoder input (P x 32 x encoder_in_feature_size), the pillar grid cells (cx, cy) and the number of occupied
    pillars before the max_voxel_size cut. rng orders the points: the first 32 of a pillar are kept."""
    points = points[rng.permutation(len(points))]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inside = ((x >= cfg.range_min[0]) & (x < cfg.range_max[0]) & (y >= cfg.range_min[1]) & (y < cfg.range_max[1])
              & (z >= cfg.range_min[2]) & (z < cfg.range_max[2]))
    points = points[inside]
    ix = np.clip(np.floor((points[:, 0] - cfg.range_min[0]) / cfg.voxel_size[0]).astype(np.int64), 0, cfg.grid_size_x - 1)
    iy = np.clip(np.floor((points[:, 1] - cfg.range_min[1]) / cfg.voxel_size[1]).astype(np.int64), 0, cfg.grid_size_y - 1)
    # the mask index of the kernel: x flipped, row-major, so the pillars go front to back
    key = (cfg.grid_size_x - 1 - ix) * cfg.grid_size_y + iy
    order = np.argsort(key, kind="stable")
    key, points = key[order], points[order]
    pillar_keys, first, counts = np.unique(key, return_index=True, return_counts=True)
    num_pillars_total = len(pillar_keys)
    num_pillars = min(num_pillars_total, cfg.max_voxel_size)

    pillar_of_point = np.repeat(np.arange(num_pillars_total), counts)
    rank = np.arange(len(key)) - first[pillar_of_point]
    kept = (rank < MAX_POINT_IN_VOXEL) & (pillar_of_point < num_pillars)
    voxels = np.zeros((num_pillars, MAX_POINT_IN_VOXEL, 4), dtype=np.float32)
    voxels[pillar_of_point[kept], rank[kept]] = points[kept]
    num_points = np.minimum(counts[:num_pillars], MAX_POINT_IN_VOXEL).astype(np.float32)
    cx = (cfg.grid_size_x - 1 - pillar_keys[:num_pillars] // cfg.grid_size_y).astype(np.int64)
    cy = (pillar_keys[:num_pillars] % cfg.grid_size_y).astype(np.int64)

    features = np.zeros((num_pillars, MAX_POINT_IN_VOXEL, cfg.encoder_in_feature_size), dtype=np.float32)
    if num_pillars == 0:
        return features, cx, cy, num_pillars_total
    mean = voxels[:, :, :3].sum(axis=1) / num_points[:, None]
    centre_x = cfg.voxel_size[0] / 2 + cx.astype(np.float32) * cfg.voxel_size[0] + cfg.range_min[0]
    centre_y = cfg.voxel_size[1] / 2 + cy.astype(np.float32) * cfg.voxel_size[1] + cfg.range_min[1]
    centre_z = cfg.voxel_size[2] / 2 + cfg.range_min[2]           # one pillar in z: coordinate 0
    features[:, :, 0:4] = voxels
    features[:, :, 4:7] = voxels[:, :, :3] - mean[:, None, :]
    features[:, :, 7] = voxels[:, :, 0] - centre_x[:, None]
    features[:, :, 8] = voxels[:, :, 1] - centre_y[:, None]
    if cfg.encoder_in_feature_size == 10:
        features[:, :, 9] = voxels[:, :, 2] - centre_z
    features[np.arange(MAX_POINT_IN_VOXEL)[None, :] >= num_points[:, None]] = 0
    return features, cx, cy, num_pillars_total


def scatter(pillar_features, cx, cy, cfg):
    """scatterFeatures_kernel: (P x 1 x C) pillar features into the 1 x C x grid_y x grid_x bird's-eye image."""
    channels = pillar_features.shape[-1]
    spatial = np.zeros((1, channels, cfg.grid_size_y, cfg.grid_size_x), dtype=np.float32)
    spatial[0][:, cy, cx] = pillar_features.reshape(-1, channels).T
    return spatial


# ------------------------------------------------------------------------------------------------------ post-process
def decode(heatmap, offset, z, dim, rot, cfg):
    """generateBoxes3D_kernel on the head outputs (1 x N x H x W): the boxes that pass the class and distance score
    thresholds and the yaw norm threshold, as an array of rows (label, score, x, y, z, length, width, height, yaw),
    highest score first."""
    score_all = 1.0 / (1.0 + np.exp(-heatmap[0]))
    label = np.argmax(score_all, axis=0)
    score = np.take_along_axis(score_all, label[None], axis=0)[0]
    yi, xi = np.indices(label.shape)
    x = cfg.voxel_size[0] * cfg.downsample_factor * (xi + offset[0, 0]) + cfg.range_min[0]
    y = cfg.voxel_size[1] * cfg.downsample_factor * (yi + offset[0, 1]) + cfg.range_min[1]
    # first distance bin whose upper limit is above the radial distance; beyond the last one: dropped
    bin_index = np.searchsorted(cfg.distance_bin_upper_limits, np.sqrt(x * x + y * y), side="right")
    in_bins = bin_index < len(cfg.distance_bin_upper_limits)
    threshold = cfg.score_thresholds[np.minimum(bin_index, len(cfg.distance_bin_upper_limits) - 1), label]
    yaw_sin, yaw_cos = rot[0, 0], rot[0, 1]
    keep = (in_bins & (score >= threshold) & (np.sqrt(yaw_sin ** 2 + yaw_cos ** 2) >= cfg.yaw_norm_thresholds[label])
            & (score > 0))
    boxes = np.stack([label[keep], score[keep], x[keep], y[keep], z[0, 0][keep], np.exp(dim[0, 1][keep]),
                      np.exp(dim[0, 0][keep]), np.exp(dim[0, 2][keep]), np.arctan2(yaw_sin[keep], yaw_cos[keep])],
                     axis=1).astype(np.float64)
    return boxes[np.argsort(-boxes[:, 1], kind="stable")]


def circle_nms(boxes, dist_threshold):
    """circleNMS: boxes sorted by score; a kept box removes the later ones within dist_threshold (any class)."""
    keep, removed = [], np.zeros(len(boxes), dtype=bool)
    threshold_sq = dist_threshold * dist_threshold
    for i in range(len(boxes)):
        if removed[i]:
            continue
        keep.append(i)
        d2 = (boxes[i + 1:, 2] - boxes[i, 2]) ** 2 + (boxes[i + 1:, 3] - boxes[i, 3]) ** 2
        removed[i + 1:] |= d2 < threshold_sq
    return boxes[keep]


def box_corners(x, y, yaw, length, width):
    """Footprint of a box (centre, yaw of the message), counter-clockwise."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((length / 2, width / 2), (-length / 2, width / 2), (-length / 2, -width / 2),
                           (length / 2, -width / 2))]


def polygon_area(polygon):
    return 0.5 * abs(sum(polygon[i][0] * polygon[i - 1][1] - polygon[i - 1][0] * polygon[i][1]
                         for i in range(len(polygon))))


def convex_intersection(subject, clip):
    """Sutherland-Hodgman: the intersection of two convex counter-clockwise polygons."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def cross_point(p, q, a, b):
        x1, y1, x2, y2, x3, y3, x4, y4 = *p, *q, *a, *b
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(d) < 1e-12:
            return q
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = list(subject)
    for i in range(len(clip)):
        a, b = clip[i - 1], clip[i]
        polygon, output = output, []
        for j in range(len(polygon)):
            p, q = polygon[j - 1], polygon[j]
            if inside(q, a, b):
                if not inside(p, a, b):
                    output.append(cross_point(p, q, a, b))
                output.append(q)
            elif inside(p, a, b):
                output.append(cross_point(p, q, a, b))
        if not output:
            break
    return output


def bev_iou(a, b):
    """compute2dIoU of perception_utils for two footprints (lists of corners)."""
    area_a, area_b = polygon_area(a), polygon_area(b)
    if area_a < 1e-6 or area_b < 1e-6:
        return 0.0
    intersection = convex_intersection(a, b)
    inter = polygon_area(intersection) if len(intersection) >= 3 else 0.0
    if inter < 1e-6:
        return 0.0
    union = area_a + area_b - inter
    return 0.0 if union < 0.01 else min(1.0, inter / union)


def iou_bev_nms(objects, search_distance_2d, iou_threshold):
    """perception_utils::IouBevNms::apply (no sorting): an object is dropped when its footprint overlaps one before it
    by more than iou_threshold. A pedestrian is only compared with pedestrians."""
    cached = []
    for obj in objects:
        p = obj.kinematics.pose_with_covariance.pose
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        cached.append((obj.classification[0].label, p.position.x, p.position.y,
                       box_corners(p.position.x, p.position.y, yaw, obj.shape.dimensions.x, obj.shape.dimensions.y)))
    search_sq = search_distance_2d * search_distance_2d
    out = []
    for t, (label_t, x_t, y_t, poly_t) in enumerate(cached):
        max_iou = 0.0
        for label_s, x_s, y_s, poly_s in cached[:t]:
            if label_t != label_s and ObjectClassification.PEDESTRIAN in (label_t, label_s):
                continue
            if (x_t - x_s) ** 2 + (y_t - y_s) ** 2 > search_sq:
                continue
            iou = bev_iou(poly_t, poly_s)
            max_iou = max(max_iou, iou)
            if iou > iou_threshold:
                break
        if max_iou <= iou_threshold:
            out.append(objects[t])
    return out


class ClassRemapper:
    """DetectionClassRemapper: relabels an object by its footprint area (detection_class_remapper.param.yaml)."""

    def __init__(self, allow, min_area, max_area):
        n = int(round(math.sqrt(len(min_area))))
        if n * n != len(min_area) or len(allow) != len(min_area) or len(max_area) != len(min_area):
            raise ValueError("class remapper matrices must be square and of the same size")
        finite_or_max = np.vectorize(lambda v: v if math.isfinite(v) else np.finfo(np.float64).max)
        self.allow = np.array(allow, dtype=bool).reshape(n, n)
        self.min_area = finite_or_max(np.array(min_area, dtype=np.float64)).reshape(n, n)
        self.max_area = finite_or_max(np.array(max_area, dtype=np.float64)).reshape(n, n)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            params = next(iter(yaml.safe_load(f).values()))["ros__parameters"]
        return cls(params["allow_remapping_by_area_matrix"], params["min_area_matrix"], params["max_area_matrix"])

    def apply(self, objects):
        for obj in objects:
            area = obj.shape.dimensions.x * obj.shape.dimensions.y
            for classification in obj.classification:
                label = classification.label
                for i in range(self.allow.shape[0]):
                    if self.allow[label, i] and self.min_area[label, i] <= area <= self.max_area[label, i]:
                        classification.label = i
                        break


def box_to_object(box, class_names):
    """box3DToDetectedObject (no twist, no variance: the v4.1 models have neither)."""
    label_index, score, x, y, z, length, width, height, yaw = box
    obj = DetectedObject()
    obj.existence_probability = float(score)
    classification = ObjectClassification()
    classification.probability = 1.0
    index = int(label_index)
    classification.label = (LABEL_OF_CLASS.get(class_names[index], ObjectClassification.UNKNOWN)
                            if 0 <= index < len(class_names) else ObjectClassification.UNKNOWN)
    if classification.label in CAR_LIKE:
        obj.kinematics.orientation_availability = DetectedObjectKinematics.SIGN_UNKNOWN
    obj.classification.append(classification)
    yaw_msg = -yaw - math.pi / 2
    pose = obj.kinematics.pose_with_covariance.pose
    pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
    pose.orientation.z, pose.orientation.w = math.sin(yaw_msg / 2), math.cos(yaw_msg / 2)
    obj.shape.type = Shape.BOUNDING_BOX
    obj.shape.dimensions.x, obj.shape.dimensions.y, obj.shape.dimensions.z = float(length), float(width), float(height)
    return obj


# ---------------------------------------------------------------------------------------------------------- detector
class CenterPointCpu:
    """CenterPointTRT::detect on the CPU: densified points (N x 4) -> boxes."""

    def __init__(self, encoder_path, head_path, cfg, num_threads=4, providers=("CPUExecutionProvider",), seed=None):
        import onnxruntime as ort  # here, so that the pre/post-processing imports without it (tests)
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder = ort.InferenceSession(encoder_path, options, providers=list(providers))
        self.head = ort.InferenceSession(head_path, options, providers=list(providers))
        self.head_outputs = [o.name for o in self.head.get_outputs()]
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def detect(self, points):
        """The boxes (rows of decode) after circle NMS, the number of occupied pillars and the step times [ms]."""
        t0 = time.perf_counter()
        features, cx, cy, num_pillars = voxelize(points, self.cfg, self.rng)
        t1 = time.perf_counter()
        if len(cx) == 0:
            return np.zeros((0, 9)), num_pillars, (1e3 * (t1 - t0), 0.0, 0.0)
        pillar_features = self.encoder.run(None, {self.encoder.get_inputs()[0].name: features})[0]
        out = dict(zip(self.head_outputs,
                       self.head.run(None, {self.head.get_inputs()[0].name: scatter(pillar_features, cx, cy, self.cfg)})))
        t2 = time.perf_counter()
        boxes = decode(out["heatmap"], out["reg"], out["height"], out["dim"], out["rot"], self.cfg)
        boxes = circle_nms(boxes, self.cfg.circle_nms_dist_threshold)
        t3 = time.perf_counter()
        return boxes, num_pillars, (1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2))


# -------------------------------------------------------------------------------------------------------------- node
XYZIRC_FIELDS = [("x", 0, PointField.FLOAT32), ("y", 4, PointField.FLOAT32), ("z", 8, PointField.FLOAT32),
                 ("intensity", 12, PointField.UINT8), ("return_type", 13, PointField.UINT8),
                 ("channel", 14, PointField.UINT16)]
XYZ_DTYPE = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3, "offsets": [0, 4, 8], "itemsize": 16})


def cloud_xyz(msg):
    """The points of a PointXYZIRC cloud (what the TensorRT node accepts, point_step 16) as N x 3, or None."""
    fields = [(f.name, f.offset, f.datatype) for f in msg.fields]
    if fields[:len(XYZIRC_FIELDS)] != XYZIRC_FIELDS or msg.point_step != 16 or msg.is_bigendian:
        return None
    raw = np.frombuffer(msg.data, dtype=XYZ_DTYPE, count=msg.width * msg.height)
    return np.stack([raw["x"], raw["y"], raw["z"]], axis=1)


def transform_matrix(t):
    """4 x 4 matrix of a geometry_msgs/Transform."""
    q, v = t.rotation, t.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), v.x],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), v.y],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), v.z],
                     [0.0, 0.0, 0.0, 1.0]])


def sweep_points(cache, cloud_capacity):
    """generateSweepPoints: the cached scans (newest first) in the frame of the newest, with their age, N x 4."""
    current_stamp, world_to_current = cache[0][1], cache[0][2]
    parts, total = [], 0
    for xyz, stamp, world_to_past in cache:
        if total + len(xyz) > cloud_capacity:
            break
        past_to_current = (world_to_current @ np.linalg.inv(world_to_past)).astype(np.float32)
        moved = xyz @ past_to_current[:3, :3].T + past_to_current[:3, 3]
        parts.append(np.hstack([moved, np.full((len(xyz), 1), current_stamp - stamp, dtype=np.float32)]))
        total += len(xyz)
    return np.vstack(parts).astype(np.float32)


class CenterPointCpuNode(Node):
    def __init__(self):
        super().__init__("lidar_centerpoint", automatically_declare_parameters_from_overrides=True)
        p = self.param
        class_names = p("model_params.class_names")
        bins = p("model_params.detection_score_thresholds.distance_bin_upper_limits")
        thresholds = {name: p(f"model_params.detection_score_thresholds.min_confidence_scores.{name}")
                      for name in class_names}
        model_range = p("model_params.point_cloud_range")
        point_cloud_range = p("cpu.point_cloud_range", model_range)
        self.cfg = Config(class_names, p("model_params.point_feature_size"), p("model_params.max_voxel_size"),
                          point_cloud_range, p("model_params.voxel_size"), p("model_params.downsample_factor"),
                          p("model_params.encoder_in_feature_size"), bins, thresholds,
                          p("post_process_params.circle_nms_dist_threshold"), p("model_params.yaw_norm_thresholds"),
                          p("cloud_capacity", 2000000))
        if p("model_params.has_twist", False) or p("model_params.has_variance", False):
            raise ValueError("models with twist or variance outputs are not supported")
        model_path = p("model_path")

        def model_file(name):
            path = p(name)
            return path if path.startswith("/") else f"{model_path}/{path}"

        self.detector = CenterPointCpu(model_file("encoder_onnx_path"), model_file("head_onnx_path"), self.cfg,
                                       p("cpu.num_threads", 4), p("cpu.providers", ["CPUExecutionProvider"]))
        self.remapper = ClassRemapper.from_yaml(model_file("class_remapper_param_path"))
        self.iou_search_distance = p("post_process_params.iou_nms_search_distance_2d")
        self.iou_threshold = p("post_process_params.iou_nms_threshold")
        self.world_frame = p("densification_params.world_frame_id")
        self.cache_size = p("densification_params.num_past_frames") + 1

        # The TF arrives through the executor of this node (main thread); the worker thread waits for it. Not
        # spin_thread=True: that adds the node to a second executor, which then also runs on_cloud.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cache = deque(maxlen=self.cache_size)      # (xyz, stamp [s], world -> scan), newest first; worker only
        self.incoming = deque(maxlen=50)                # (header, xyz) received, not yet in the cache
        self.condition = threading.Condition()
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.publisher = self.create_publisher(DetectedObjects, "~/output/objects", QoSProfile(depth=1))
        self.create_subscription(PointCloud2, "~/input/pointcloud", self.on_cloud, qos)
        self.stats = {"received": 0, "processed": 0, "times": []}
        self.create_timer(10.0, self.report)
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f"CenterPoint on the CPU ({', '.join(self.detector.encoder.get_providers())}, {p('cpu.num_threads', 4)} "
            f"threads): range x {point_cloud_range[0]:g}..{point_cloud_range[3]:g} y {point_cloud_range[1]:g}.."
            f"{point_cloud_range[4]:g} m (model {model_range[3]:g} m), grid {self.cfg.grid_size_x} x "
            f"{self.cfg.grid_size_y}, {self.cache_size} scan(s) per detection")

    def param(self, name, default=None):
        if self.has_parameter(name):
            return self.get_parameter(name).value
        if default is None:
            raise ValueError(f"parameter {name} missing (ml_package.param.yaml / centerpoint_common.param.yaml)")
        return default

    def on_cloud(self, msg):
        """Executor thread: only queues the scan (the TF lookup may wait, see run)."""
        xyz = cloud_xyz(msg)
        if xyz is None:
            self.get_logger().error("expected PointXYZIRC clouds (point_step 16): skipped", once=True)
            return
        with self.condition:
            self.incoming.append((msg.header, xyz))
            self.stats["received"] += 1
            self.condition.notify()

    def enqueue(self, header, xyz):
        """PointCloudDensification::enqueuePointCloud: the scan into the cache with its world -> scan transform."""
        if self.cache_size > 1:
            try:
                tf = self.tf_buffer.lookup_transform(header.frame_id, self.world_frame, Time.from_msg(header.stamp),
                                                     timeout=Duration(seconds=0.5))
            except TransformException as e:
                self.get_logger().warning(str(e), throttle_duration_sec=5.0)
                return False
            world_to_current = transform_matrix(tf.transform)
        else:
            world_to_current = np.eye(4)
        self.cache.appendleft((xyz, header.stamp.sec + header.stamp.nanosec * 1e-9, world_to_current))
        return True

    def run(self):
        """Worker thread: every scan goes into the cache (densification), only the newest is detected."""
        while rclpy.ok():
            with self.condition:
                while not self.incoming and rclpy.ok():
                    self.condition.wait(timeout=0.5)
                scans = list(self.incoming)
                self.incoming.clear()
            header = None
            for scan_header, xyz in scans:
                if self.enqueue(scan_header, xyz):
                    header = scan_header
            # nothing new in the cache, or (as the TensorRT node) nobody listening
            if header is None or self.publisher.get_subscription_count() == 0:
                continue
            start = time.perf_counter()
            boxes, num_pillars, step_times = self.detector.detect(sweep_points(list(self.cache),
                                                                               self.cfg.cloud_capacity))
            if num_pillars > self.cfg.max_voxel_size:
                self.get_logger().warning(f"{num_pillars} pillars, more than max_voxel_size "
                                          f"{self.cfg.max_voxel_size}", throttle_duration_sec=1.0)
            objects = iou_bev_nms([box_to_object(b, self.cfg.class_names) for b in boxes],
                                  self.iou_search_distance, self.iou_threshold)
            self.remapper.apply(objects)
            out = DetectedObjects()
            out.header = header
            out.objects = objects
            self.publisher.publish(out)
            self.stats["processed"] += 1
            self.stats["times"].append((1e3 * (time.perf_counter() - start), *step_times))

    def report(self):
        received, processed, times = self.stats["received"], self.stats["processed"], self.stats["times"]
        self.stats = {"received": 0, "processed": 0, "times": []}
        if not times:
            if received:
                self.get_logger().info(f"{received} scans in 10 s, none processed (no subscriber?)")
            return
        t = np.array(times)
        self.get_logger().info(
            f"{processed}/{received} scans in 10 s, {np.median(t[:, 0]):.0f} ms median ({np.max(t[:, 0]):.0f} max): "
            f"pillars {np.median(t[:, 1]):.0f}, networks {np.median(t[:, 2]):.0f}, decode {np.median(t[:, 3]):.0f} ms")


def main():
    rclpy.init()
    node = CenterPointCpuNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
