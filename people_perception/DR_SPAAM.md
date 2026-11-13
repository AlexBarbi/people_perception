# DR-SPAAM CPU pipeline

The default detector (`dr_spaam_cpu`, also selected by `auto`) is:

```text
/rslidar_points -> pointcloud_to_laserscan -> /perception/scan (base_link)
 -> dr_spaam_cpu.py -> /perception/detector/objects (DetectedObjects)
 -> people_tracker -> /perception/tracks (camera_init)
 -> obstacle_adapter -> /perception/dynamic_obstacles (DynamicObstacleArray)
```

The tracker and adapter message contracts are unchanged. Classification is
PEDESTRIAN; existence probability is sigmoid(logit). Both empty and nonempty
results retain the input scan stamp and frame. The converter retains the cloud
stamp, with frame changed by TF. No wall-clock stamp or scan-time shift is added.
Inference failures do not publish a synthetic empty scene. Sensor QoS and depth
one retain only the latest waiting scan during inference.

## Dependencies and build

```bash
sudo apt install ros-jazzy-pointcloud-to-laserscan
python3 -m venv --system-site-packages .venv-dr-spaam
.venv-dr-spaam/bin/python -m pip install numpy onnxruntime
source /opt/ros/jazzy/setup.bash
colcon build --packages-select people_perception_msgs people_perception
source install/setup.bash
```

`python3-onnxruntime-pip` is the rosdep key for the CPU runtime. The launch `python`
argument must select an interpreter compatible with the installed ROS Python ABI.
PyTorch and the upstream package are required only for export, not on the robot.
CenterPoint remains available by explicit selection with its optional dependencies.

## Export the official model

Supported upstream: [RWTHVision/DR-SPAAM-Detector](https://github.com/RWTHVision/DR-SPAAM-Detector),
`SpatialDROW`, release v1.1 `dr_spaam_e40.pth`, checkpoint key `model_state`.
The newer `2D_lidar_person_detection` JRDB architecture/checkpoints are different
and are not accepted. Respect the upstream code/weights license (the upstream
reference repository uses GPL-3.0); weights are not bundled with this package.

```bash
git clone https://github.com/RWTHVision/DR-SPAAM-Detector.git /tmp/DR-SPAAM-Detector
.venv-dr-spaam/bin/python -m pip install torch onnx
curl -L --fail https://github.com/RWTHVision/DR-SPAAM-Detector/releases/download/v1.1/dr_spaam_e40.pth -o /tmp/dr_spaam_e40.pth
mkdir -p models/dr_spaam
PYTHONPATH=/tmp/DR-SPAAM-Detector/dr_spaam/src \
  .venv-dr-spaam/bin/python src/people_perception/people_perception/scripts/export_dr_spaam_onnx.py \
  /tmp/dr_spaam_e40.pth models/dr_spaam/dr_spaam.onnx --stride 1
```

Exporter checks official PyTorch versus local attention, then PyTorch versus CPU
ONNX over initialization, recurrent steps and two scan sizes. ONNX inputs are
`cutouts: float32[1,N,1,56]`, `template: float32[1,N,256,14]`,
`first_scan: bool[1]`; outputs are `logits[1,N,1]`, `offsets[1,N,2]`,
`template_out[1,N,256,14]`. N is dynamic. Runtime checks contract metadata and
matching stride. State resets on changed frame/grid, backwards/equal timestamps,
gaps over 0.5 s, all-invalid scans or inference errors.

Preprocessing follows the official 1 m wide, 0.5 m deep, 56-point cutouts with
area sampling and 29.99 m padding. Invalid rays are padded but cannot vote.
Offsets are tangential and radial displacements, not absolute person coordinates:
`x=(r+dy)*cos(theta)-dx*sin(theta)`,
`y=(r+dy)*sin(theta)+dx*cos(theta)`. This uses the actual scan angle_min,
including asymmetric sectors. NMS suppresses centers closer than 0.5 m.
Local attention evaluates the same eleven neighbors as the official binary mask,
without its quadratic dense similarity/feature aggregation.

## TF, heights and launch

Configure the measured LiDAR pose in URDF/TF. Do not assume the example mounting
height 0.45 m: it was not supplied. Required TF paths at each cloud stamp:
`base_link <- <cloud frame>` and `camera_init <- base_link`.
`base_link` must have z=0 at the floor; if the existing robot base origin is
above the floor, use a calibrated floor-level frame via `scan_frame` instead.
The current FAST-LIO integration uses `lio_base`: it must be connected correctly
to this floor frame. Merely changing a frame name does not supply that transform.

The height band 0.75–1.05 m is evaluated **after** transforming the cloud.
At an actual 0.45 m LiDAR height, this means approximately 0.30–0.60 m above
the sensor for a level mount. Do not subtract mounting height again in YAML.
The selected band excludes floor/ceiling; shelving at that height still appears
in the scan and must be rejected by the learned detector.

```bash
ros2 launch people_perception people_perception_launch.xml \
  people:=true use_sim_time:=false \
  python:="$PWD/.venv-dr-spaam/bin/python" \
  dr_spaam_model:="$PWD/models/dr_spaam/dr_spaam.onnx" \
  lidar_topic:=/rslidar_points scan_frame:=base_link
```

When the parent FAST-LIO launch already runs this chain, do not launch another
copy. Its `detector:=auto` selects DR-SPAAM; `perception_python`, `dr_spaam_model`,
`scan_frame`, and `dr_spaam_threads` are forwarded to this package.
For the existing flat-floor quadruped simulation, `perception_floor_tf.py`
publishes `odom -> perception_floor` from `/odom` x/y/yaw at a fixed `floor_z`
(default 0). Run it with `use_sim_time:=true` and select
`scan_frame:=perception_floor`. It removes body roll/pitch/height from the slice
frame and leaves the existing `base` frame unchanged. This assumes a known flat
floor in the odometry coordinate system; it is not terrain estimation.

Default sector is +/-110 degrees. Full scan:
`scan_angle_min:=-3.141592653589793 scan_angle_max:=3.141592653589793`.
Cutout/attention edges use padding and clipped neighborhoods, matching the
training implementation; a full scan does not add circular attention across
the +/-pi seam. YAML fixes increment at 0.00785 rad, range 0.3–15 m. Official
DROW example uses 0.5 degrees; 0.45 degrees here is the requested deployment grid,
not a claim about all pretrained scanners. Adjust scan_time to the driver rate.

The nominal person box is 0.6 x 0.6 x 1.7 m, centered 0.85 m above the floor,
with unknown heading. This preserves full-body cloud filtering despite 2D
inference. Adapter radius is 0.4 m with the existing 0.1 m margin and 0.3–0.5 m
clamps; maximum speed remains 1.5 m/s. Adapter range is 15 m but the existing
controller still accepts only four closest obstacles. Tracker uses score 0.5
and initial position noise 0.2 m; these are initial tuning values.

## Validation and performance limits

Build and five ROS-message/geometry/state tests passed on ROS 2 Jazzy. Official
cutout parity was checked at 490/801 beams and strides 1/2 (tolerance 2e-4;
beam-index arithmetic differs slightly from upstream angle arithmetic). Export
was checked with actual official v1.1 weights, not random placeholder weights.

Measured in this development environment, CPUExecutionProvider, four threads,
FP32, stride 1, 30 timed scans after five warmups:

| Beams | Preprocess + ONNX + decode median | p95 |
|---|---:|---:|
| 490 (+/-110 degrees) | 180 ms | 197 ms |
| 801 (approximately 360 degrees) | 286 ms | 325 ms |

These synthetic scan measurements exclude cloud projection, TF, transport and
ROS serialization. **The requested <15 ms target is not achieved.** Python
vectorization and local attention do not remove the convolutional network cost.
Measure on the actual robot with recorded data; narrow the sector or increase
stride only after validating recall. To change stride, re-export with `--stride K`
and launch with `dr_spaam_stride:=K`. This changes effective angular spacing,
attention support and detection recall. Runtime warns when callback exceeds its
15 ms budget; this budget is a diagnostic threshold, not a scheduling guarantee.

Reproduce timing with the ROS-independent benchmark:

```bash
.venv-dr-spaam/bin/python src/people_perception/people_perception/scripts/benchmark_dr_spaam.py \
  models/dr_spaam/dr_spaam.onnx --beams 490 --threads 4 --scans 30
```

The official scanner example was mounted around 0.37 m, whereas the requested
projection observes 0.75–1.05 m. Person recall and shelving false positives cannot
be inferred from model provenance. Validate the height band, confidence, sparse
Airy returns, motion distortion, latency and aisle occlusions on warehouse bags;
fine-tuning may be necessary. Raw collision-avoidance cloud consumers remain
independent of learned person classification.
