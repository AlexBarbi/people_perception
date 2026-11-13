"""NumPy preprocessing and recurrent ONNX contract for official DR-SPAAM-Detector weights.

Reference: RWTHVision/DR-SPAAM-Detector, detector.py and utils/utils.py.
No PyTorch or ROS dependency is needed at runtime.
"""
import math

import numpy as np


def cutouts(scan, angle_increment, stride=1):
    """Single-scan, 1 m wide / 0.5 m deep, 56-point area-sampled cutouts.

    Same normalization and boundary padding as the official detector. Work in
    beam indices to support asymmetric LaserScan sectors without rotating data.
    """
    centers = np.arange(0, len(scan), stride)
    distances = scan[centers]
    half_width = np.arctan(0.5 / np.maximum(distances, 1e-2)).astype(np.float64) / angle_increment
    indices = centers[None, :] + np.linspace(-1, 1, 56)[:, None] * half_width
    outside = (indices < 0) | (indices > len(scan) - 1)
    low = np.clip(np.floor(indices), 0, len(scan) - 1).astype(np.intp)
    high = np.minimum(low + 1, len(scan) - 1)
    ratio = np.clip(indices - low, 0, 1)
    sampled = scan[low] + ratio * (scan[high] - scan[low])
    wide = 2 * half_width > 56
    if np.any(wide):
        factor = int(math.ceil(np.max(2 * half_width) / 56))
        area_indices = (centers[None, wide] +
                        np.linspace(-1, 1, 56 * factor)[:, None] * half_width[None, wide])
        area_indices = np.rint(np.clip(area_indices, 0, len(scan) - 1)).astype(np.intp)
        sampled[:, wide] = scan[area_indices].reshape(56, factor, -1).mean(axis=1)
    sampled[outside] = 29.99
    sampled = (np.clip(sampled, distances - 0.5, distances + 0.5) - distances) / 0.5
    return np.ascontiguousarray(sampled.T[None, :, None, :], dtype=np.float32)


def decode(scan, angles, valid, logits, offsets, threshold=0.5, nms_distance=0.5,
           range_min=0.3, range_max=15.0):
    """Convert canonical (tangential, radial) offsets into ROS x-forward/y-left.

    Invalid source beams cannot vote. NMS runs only on confident finite votes,
    with O(N) temporary memory instead of the upstream dense distance matrix.
    """
    logits = np.asarray(logits).reshape(-1)
    offsets = np.asarray(offsets).reshape(-1, 2)
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    radial = scan + offsets[:, 1]
    x = radial * np.cos(angles) - offsets[:, 0] * np.sin(angles)
    y = radial * np.sin(angles) + offsets[:, 0] * np.cos(angles)
    xy = np.column_stack((x, y))
    radius = np.hypot(x, y)
    candidates = np.flatnonzero(valid & np.isfinite(xy).all(axis=1) &
                               np.isfinite(scores) & (scores >= threshold) &
                               (radius >= range_min) & (radius <= range_max))
    order = candidates[np.argsort(-scores[candidates], kind='stable')]
    kept = []
    while order.size:
        best = order[0]
        kept.append(best)
        rest = order[1:]
        distance_sq = np.sum((xy[rest] - xy[best]) ** 2, axis=1)
        order = rest[distance_sq >= nms_distance ** 2]
    return xy[kept], scores[kept]


class OnnxDetector:
    """Explicit recurrent state; reset on frame/grid changes and time gaps."""

    def __init__(self, model_path, threads=4, stride=1):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=options,
                                            providers=['CPUExecutionProvider'])
        metadata = self.session.get_modelmeta().custom_metadata_map
        if metadata.get('people_perception.dr_spaam') != '1':
            raise ValueError('Use export_dr_spaam_onnx.py: incompatible ONNX model contract')
        if int(metadata['stride']) != stride:
            raise ValueError('Runtime stride must match export --stride (attention spacing)')
        if {i.name for i in self.session.get_inputs()} != {'cutouts', 'template', 'first_scan'}:
            raise ValueError('Unexpected DR-SPAAM ONNX inputs')
        self.stride = stride
        self.template = None

    def reset(self):
        self.template = None

    def infer(self, scan, angle_increment):
        data = cutouts(scan, angle_increment, self.stride)
        count = data.shape[1]
        first = self.template is None
        if first:
            self.template = np.zeros((1, count, 256, 14), dtype=np.float32)
        logits, offsets, template = self.session.run(
            ['logits', 'offsets', 'template_out'],
            {'cutouts': data, 'template': self.template,
             'first_scan': np.array([first], dtype=np.bool_)})
        if not all(np.isfinite(value).all() for value in (logits, offsets, template)):
            self.reset()
            raise ValueError('Non-finite ONNX output; recurrent state cleared')
        self.template = template
        return logits, offsets
