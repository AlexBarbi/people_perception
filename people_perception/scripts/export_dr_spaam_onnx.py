#!/usr/bin/env python3
"""Export official RWTHVision/DR-SPAAM-Detector SpatialDROW checkpoints.

Install the official repository's dr_spaam package plus torch, onnx and
onnxruntime in an export environment. PyTorch is not needed on the robot.
This adapter supports dr_spaam_e40.pth (model_state), not the newer JRDB models.
Upstream code/weights retain their own license (GPL-3.0 for the reference code).
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from dr_spaam.model.drow import SpatialDROW


class ExportStep(nn.Module):
    """Exact local spatial attention with O(N * 11) rather than O(N squared) pairs."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, cutouts, template, first_scan):
        current = self.model._forward_cutout(cutouts).squeeze(2)
        count = current.shape[1]
        emb = self.model.gate.conv(current.reshape(-1, 256, 14)).reshape(1, count, 128)
        prev_emb = self.model.gate.conv(template.reshape(-1, 256, 14)).reshape(1, count, 128)
        # Mask out-of-bounds neighbors instead of clamping duplicates, matching
        # upstream's binary neighbor mask at sector edges.
        indices = (torch.arange(count, device=current.device)[:, None] +
                   torch.arange(-5, 6, device=current.device)[None, :])
        valid = (indices >= 0) & (indices < count)
        indices = indices.clamp(0, count - 1)
        similarity = (emb[:, :, None, :] * prev_emb[:, indices, :]).sum(-1)
        similarity = similarity.masked_fill(~valid[None, :, :], -1e10)
        attention = torch.softmax(similarity, dim=-1)
        # Accumulate one neighbor at a time. Avoid N*11*256*14 feature temporary.
        weighted = torch.zeros_like(current)
        for neighbor in range(11):
            weighted = weighted + (attention[:, :, neighbor, None, None] *
                                   template[:, indices[:, neighbor], :, :])
        fused = self.model.gate._alpha * current + (1 - self.model.gate._alpha) * weighted
        fused = torch.where(first_scan.reshape(1, 1, 1, 1), current, fused)
        logits, offsets = self.model._forward_fused_cutout(fused)
        return logits, offsets, fused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--beams', type=int, default=490, help='Example scan size; exported axis is dynamic')
    parser.add_argument('--stride', type=int, default=1, help='Must match runtime stride')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.beams < 2 or args.stride < 1 or args.threads < 1:
        parser.error('Require beams >= 2, stride >= 1 and threads >= 1')
    torch.set_num_threads(args.threads)
    torch.manual_seed(7)
    model = SpatialDROW(num_pts=56, pedestrian_only=True, alpha=0.5, window_size=11).cpu().eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    wrapper = ExportStep(model).eval()
    count = len(range(0, args.beams, args.stride))
    data = torch.randn(1, count, 1, 56)
    template = torch.randn(1, count, 256, 14)
    first = torch.tensor([False])
    # Validate the CPU optimization against the official full attention, including
    # first-frame initialization. No loss of recurrent semantics during tracing.
    with torch.no_grad():
        for initialize in (True, False):
            official = model(data, testing=True, fea_template=None if initialize else template)
            local = wrapper(data, template, torch.tensor([initialize]))
            for actual, expected in zip(local, official[:3]):
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        torch.onnx.export(
            wrapper, (data, template, first), str(args.output), opset_version=17, dynamo=False,
            input_names=['cutouts', 'template', 'first_scan'],
            output_names=['logits', 'offsets', 'template_out'],
            dynamic_axes={name: {1: 'cutouts_count'} for name in
                          ('cutouts', 'template', 'logits', 'offsets', 'template_out')})
    graph = onnx.load(str(args.output))
    onnx.helper.set_model_props(graph, {'people_perception.dr_spaam': '1', 'stride': str(args.stride),
                                       'source': 'RWTHVision/DR-SPAAM-Detector SpatialDROW',
                                       'cutout_points': '56', 'padding_val': '29.99'})
    onnx.checker.check_model(graph)
    onnx.save(graph, str(args.output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
    # Check ONNX parity across initialization, recurrent updates and changed beam
    # counts: tracing must not silently freeze the attention neighborhood shape.
    with torch.no_grad():
        for n in (count, max(2, count // 2)):
            data = torch.randn(1, n, 1, 56)
            template = torch.zeros(1, n, 256, 14)
            for step in range(3):
                first = torch.tensor([step == 0])
                expected = wrapper(data, template, first)
                actual = session.run(None, {'cutouts': data.numpy(), 'template': template.numpy(),
                                            'first_scan': first.numpy()})
                for a, e in zip(actual, expected):
                    np.testing.assert_allclose(a, e.numpy(), atol=2e-4, rtol=2e-4)
                template = expected[2]
    print(f'Exported and checked recurrent CPU ONNX: {args.output}')


if __name__ == '__main__':
    main()
