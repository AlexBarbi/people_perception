# people_perception

People detection and tracking on the lidar scans of FAST-LIO, for a quadruped that explores a warehouse with a
RoboSense Airy. It does three things with the people it tracks:

1. keeps their points **out of the map** SC-PGO builds,
2. keeps their points **out of the exploration coverage** of TARE,
3. hands them to the **Opti-Pessi MPC controller** as dynamic obstacles.

One package, `people_perception`: three C++ nodes, CenterPoint on the CPU (`scripts/centerpoint_cpu.py`), the launch
file, the config files, the tests and a dataset tool.

## Two clouds from every scan

The same scan goes to two kinds of consumers, and only one of them gets it filtered:

- **Full cloud**, people included: everything that avoids collisions (`terrainAnalysis`, `terrainAnalysisExt`,
  `far_planner`). These keep reading `/cloud_registered` from FAST-LIO.
- **Static cloud**, people removed: everything that accumulates scans over time (TARE coverage, SC-PGO keyframes).

The static cloud must never feed what avoids collisions. A detector mistake can then dirty a map, not cause a hit.
For the same reason the filter lets the clouds through unfiltered when the detector is down.

## Data flow

```
                     FAST-LIO
        /cloud_registered_body + /Odometry
                        │
                        ▼
              detector_cloud_adapter
                        │  /perception/detector/pointcloud   (sensor_level, PointXYZIRC)
                        ▼
  autoware_lidar_centerpoint  or  centerpoint_cpu.py
                        │  /perception/detector/objects      (sensor_level)
                        ▼
                 people_tracker
              │                       │
   /opti_pessi/obstacles      /perception/tracks             (camera_init)
   (odom, confirmed people)           │
              │                       ▼
              ▼              dynamic_points_filter
   OptiPessiController         /cloud_registered      ─► /registered_scan_static        ─► TARE
                               /cloud_registered_body ─► /cloud_registered_body_static  ─► SC-PGO
                                                         /perception/dynamic_points     ─► rviz
```

FAST-LIO stamps the two clouds, the odometry and the TF of a scan alike (the end of the scan). Every node here keeps
that stamp unchanged: it is the key that pairs clouds, detections and tracks.

## Running it

The launch file is included by `fast_lio_slam_launch.xml` of the `ranav_bringup` package, which forwards the
arguments below. **Do not launch `people_perception_launch.xml` by hand next to it**: a second filter would double
every scan.

```bash
ros2 launch ranav_bringup fast_lio_slam_launch.xml people:=true
```

| Argument | Default | Effect |
|---|---|---|
| `people` | `false` | `false`: only `dynamic_points_filter`, as a pass-through. `true`: the whole chain, and the filter removes the tracked people |
| `centerpoint_model` | `models/lidar_centerpoint/v4.1/short_range` of the workspace | Model folder of the detector |
| `detector` | `auto` | `centerpoint` (TensorRT), `centerpoint_cpu` (ONNX Runtime) or `auto`: `centerpoint` when `autoware_lidar_centerpoint` is installed, else `centerpoint_cpu` |
| `opti_pessi_obstacles` | `true` | The confirmed people go to `/opti_pessi/obstacles`. `false` when another node publishes that topic |
| `use_sim_time` | `true` | |

SC-PGO and TARE read their scans from `dynamic_points_filter` in both modes: **if the filter is not running, neither
gets any scan.** `people:=true` starts the detector: with CUDA and TensorRT the Autoware one, without them (not built)
the same model on the CPU.

## Nodes

No parameter has a default in the code: each node reads everything from its file in `config/` and stops if one is
missing. The files carry the meaning and the unit of every value.

### `detector_cloud_adapter`

Turns the body cloud of FAST-LIO into what a CenterPoint node takes.

- **Layout**: `PointXYZIRC` of `autoware_point_types`, 16 bytes (`x y z` FLOAT32, `intensity` UINT8, `return_type`
  UINT8, `channel` UINT16). The model reads `x y z` only: `intensity`, `return_type` and `channel` are 0.
- **Frame** `sensor_level`: origin and yaw of the sensor, no roll, no pitch. A pillar detector assumes a level floor
  at a constant height, and the body of a trotting quadruped rolls and pitches. The node publishes the TF
  `camera_init -> sensor_level` with every cloud; the tracker uses it to bring the detections back.
- **Crop**: the robot itself (`self_filter_radius`), `max_range_xy`, `z_min .. z_max`. A detector has to be trained on
  clouds cropped the same way.
- A cloud is paired with the odometry message of the same stamp. The small odometry message can arrive after the next
  cloud, so clouds wait for theirs in order (`cloud_queue`); a scan is dropped only when a later odometry is here and
  its own is not.

### Detector: `autoware_lidar_centerpoint` or `centerpoint_cpu.py`

The CenterPoint node of Autoware Universe (not part of this repository), started by the launch file. It answers every
cloud on `/perception/detector/pointcloud` with one `autoware_perception_msgs/DetectedObjects` on
`/perception/detector/objects`, same header. Its parameters are in `config/centerpoint_common.param.yaml` (Autoware's
file with `world_frame_id: camera_init` and a tighter NMS).

- Needs CUDA and TensorRT. In the RaNAV workspace: CUDA toolkit 13.0 and TensorRT 10.13 (`./setup_ranav.sh cuda` at the
  workspace root), `LD_LIBRARY_PATH` including `/usr/local/cuda-13.0/lib64`.
- `centerpoint_model` is a folder with the two `.onnx`, `ml_package.param.yaml` and
  `detection_class_remapper.param.yaml`. The TensorRT engines are built there on the first start (minutes) and are not
  portable between machines or TensorRT versions.
- The pretrained Autoware models were trained on road data: **a proof that the chain runs, not a people detector for a
  warehouse.**

`scripts/centerpoint_cpu.py` is the same model, the same parameter files and the same topics with ONNX Runtime on the
CPU, for a computer without an NVIDIA GPU, on a smaller area (`config/centerpoint_cpu.param.yaml`: about 84 ms per scan
at +-12.8 m). Scans that come while it is busy are only used for the densification: those scans get no detections.

Measured with the pretrained `v4.1/short_range` model on the two walking actors of the Gazebo warehouse, TensorRT,
headless, 51 s:

| | |
|---|---|
| A track on an actor | 97 % of the time |
| Tracks on nobody | 2 % |
| Actor points removed | 99 % |
| Confirmed tracks started for 2 people | 8 |
| Inference | 45 ms median, 100 ms p95 |
| GPU memory, with the simulation | 2.3 GB |

### `people_tracker`

Follows the people from scan to scan: it smooths their position, estimates their velocity, and keeps them for a while
when the detector misses them. The confirmed ones are the obstacles of the controller.

- Takes the `PEDESTRIAN` detections only and brings them to `camera_init` with the TF at their stamp. That frame
  drifts slowly but never jumps; the frame SC-PGO corrects jumps at every loop closure and would break the tracks.
- One Kalman filter per person (position and velocity on the plane, constant velocity). Detections are matched to
  tracks by the smallest total distance (Hungarian algorithm), within `gate_distance` and `max_plausible_speed`.
- A track starts **tentative**, is **confirmed** after `min_hits` detections in `hits_window` scans with a mean score
  of `min_score_confirm`, **coasts** on its prediction while no detection matches, and is dropped after `max_age`.
- Publishes one message on each topic per message received, an empty one included: the filter reads it as "this scan
  is done", and the controller clears its obstacles on an empty array.

| Topic | Content |
|---|---|
| `/perception/tracks` | Every track, confirmed, tentative and coasting: what the filter removes |
| `/perception/tracks_markers` | Box (green confirmed, yellow tentative, faint coasting), speed, velocity arrow |
| `/opti_pessi/obstacles` | `legged_controllers/ObstacleArray`, frame `odom`: the confirmed tracks within `obstacles.max_range` of the sensor, as `HUMAN` |

In a `TrackedObject`, `object_id` is left empty (the tracks carry no identity), the pose has no heading,
`twist.linear` is the velocity in `camera_init` (not in the frame of the object, as Autoware has it), and the x y
blocks of the pose and twist covariances are those of the Kalman filter.

**The obstacles of the controller.** `OptiPessiController` reads `/opti_pessi/obstacles` in `odom` only and does no
TF lookup.

- The positions are brought to `odom` with the latest TF `odom -> lio_map -> camera_init`, which FAST-LIO publishes
  (`fast_lio_slam_launch.xml`, parameters `alignment.*`) once the legged estimator's `/odom` runs. Until then nothing
  is published, not an empty array: that would tell the controller there is nobody.
- Only type and position reach the controller. Radius and maximum speed are those of `HUMAN` in
  `opti_pessi_interface/config/task.info`, and the controller keeps the closest obstacles for its slots.
- `obstacles.max_range` (10 m): each obstacle takes a slot of the controller, and the solve time grows with them.
- Each message replaces the previous one. **Two publishers erase each other**: do not run the camera detector
  `yolo26_rgbd_detect` next to it, or pass `opti_pessi_obstacles:=false`.

### `dynamic_points_filter`

Stands between FAST-LIO and what accumulates its scans.

| Input | Output | Reader |
|---|---|---|
| `/cloud_registered` (`camera_init`) | `/registered_scan_static` | TARE; `far_static/terrainAnalysis(Ext)` -> `/terrain_map_static`, `/terrain_map_ext_static` -> far_planner (exploration: the saved graph; inspections: its routes, unless `people_free:=false`) |
| `/cloud_registered_body` (`rslidar_lio`) | `/cloud_registered_body_static` | SC-PGO |
| | `/perception/dynamic_points` | what was taken out of `/cloud_registered`, to look at |

- The outputs keep the fields, the `point_step`, the frame and the stamp of the inputs; the points that stay keep every
  byte.
- `enabled: false` (`people:=false`): every cloud goes out as it came in, at once.
- `enabled: true`: a cloud waits for the tracks of its scan. Around each track of `/perception/tracks` a vertical
  cylinder is cut out: the box plus margins, plus `position_sigma_factor` times the position uncertainty of the track,
  which grows while it coasts. The body cloud is tested through the odometry of its scan.
- A cloud without the tracks of its scan (they did not come within `max_wait`, or the detector skipped the scan) goes
  with the tracks of the closest scan, moved on with their velocity, if within `max_track_staleness`; unfiltered
  otherwise.
- **The clouds of a topic go out in scan order**, whatever each one waits for. SC-PGO pairs each cloud with the next
  odometry in its queue: one cloud doubled shifts every later pair and ruins the map.
- **One filter only**: the node waits `duplicate_check_wait` after its start and stops if an output is already
  published.
- `/diagnostics`: clouds filtered with the tracks of their scan, with those of another scan, clouds unfiltered, points
  removed, longest wait. The status is WARN only when clouds went out unfiltered (detector down).

## Build

```bash
colcon build --packages-select people_perception
```

| Dependency | Where it comes from |
|---|---|
| `rclcpp`, `sensor_msgs`, `std_msgs`, `nav_msgs`, `geometry_msgs`, `visualization_msgs`, `tf2`, `tf2_ros`, `tf2_eigen`, `diagnostic_updater`, Eigen | ROS 2 Jazzy |
| `autoware_perception_msgs` | In the RaNAV workspace built from `third_party/autoware_msgs` (tag 1.13.0, that package only) |
| `legged_controllers` | Its messages only (`Obstacle`, `ObstacleArray`): `CMakeLists.txt` imports their targets without `find_package(legged_controllers)`, which would load the whole controller stack (ocs2, pinocchio, casadi) |
| FAST-LIO | Run time: `/cloud_registered`, `/cloud_registered_body`, `/Odometry` with the same stamp per scan |
| `autoware_lidar_centerpoint` | Run time, with `people:=true` and `detector:=centerpoint` |
| `python3-numpy`, `python3-yaml`, `rclpy`, `tf2_ros_py`, `rosbag2_py`, `rosidl_runtime_py` | The Python scripts; `centerpoint_cpu.py` also needs `onnxruntime` (pip, the `.venv` of the workspace) |

Config and launch files are installed as copies: an edit does nothing until the package is rebuilt.

## Tests and checks

```bash
colcon test --packages-select people_perception --event-handlers console_direct+
```

The node tests run the nodes in process on a `ROS_DOMAIN_ID` of their own, away from a running stack.

With the stack running (FAST-LIO and what `fast_lio_slam_launch.xml` starts with it); paths are relative to this
repository:

```bash
# the static clouds are the FAST-LIO ones byte for byte, none lost, none twice; same stamp on both clouds and /Odometry
python3 test/dynamic_points_filter_check.py

# with people:=true: the same points in the same order but those taken out; /perception/dynamic_points is exactly what
# is missing; a detector cloud and tracks for every scan (with the CPU detector the scans it skipped fail "tracks")
python3 test/dynamic_points_filter_check.py --people

# what the filter did since the last report
ros2 topic echo /diagnostics | grep -A12 dynamic_points_filter
```

## Dataset for a detector model

`scripts/export_detector_dataset.py` turns a bag into the raw data of a mmdetection3d custom dataset, made of exactly
the clouds the detector node receives.

```bash
ros2 bag record -o <bag> /perception/detector/pointcloud /perception/detector/objects /tf /tf_static /Odometry
ros2 run people_perception export_detector_dataset.py <bag> <out_dir> --every-seconds 0.5
```

| Output | |
|---|---|
| `points/000000.bin` | float32, N x 4: `x y z time_lag` (`time_lag` is 0) |
| `labels/000000.txt` | One box per line: `x y z dx dy dz yaw Pedestrian`, `x y z` the centre of the bottom face |
| `ImageSets/train.txt`, `val.txt` | Validation is the last `--val-fraction` of the bag |
| `frames.csv` | Per frame: stamp, pose of the cloud frame in the odometry frame, number of points and of labels |

The labels are the answers of the detector recorded in the bag: a first guess to correct by hand. Without that topic
in the bag there is no `labels/` folder and the frames are to be annotated.

## Limits

- Pedestrians only. Other classes of a detector are ignored by the tracker.
- No model trained on warehouse data yet: with the pretrained CenterPoint a person is often lost and tracked anew
  (8 tracks for 2 people in 51 s).
- The tracks carry no identity: nothing tells that the person of one scan is the person of the next, beyond the
  track itself going on.
- The first SC-PGO keyframe is the first scan it gets, before any detector has answered: its people stay in the map.
- The controller takes position and type only, and the lidar people and the camera detector cannot both publish
  `/opti_pessi/obstacles`.
- The margins of the filter, the tracker tuning and the obstacle range are starting values, to be tuned on recorded
  data.
