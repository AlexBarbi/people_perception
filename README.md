# people_perception

People detection and tracking on the lidar scans of FAST-LIO, for a quadruped that explores a warehouse with a
RoboSense Airy. It does three things with the people it tracks:

1. keeps their points **out of the map** SC-PGO builds,
2. keeps their points **out of the exploration coverage** of TARE,
3. hands them to the **Opti-Pessi MPC controller** as dynamic obstacles.

The design is the one of `docs/perception_pipeline.md` in the RaNAV workspace (not in this repository); the block
names B3, B5, B6, B10 in the code and in the config files come from there.

| Package | Content |
|---|---|
| `people_perception` | The four nodes, the launch file, the config files, the tests and tools |
| `people_perception_msgs` | `DynamicObstacle`, `DynamicObstacleArray`: the tracked people as obstacles for a local planner |

## Two clouds from every scan

The same scan goes to two kinds of consumers, and only one of them gets it filtered:

- **Full cloud**, people included: everything that avoids collisions (`terrainAnalysis`, `terrainAnalysisExt`,
  `far_planner`). These keep reading `/cloud_registered` from FAST-LIO.
- **Static cloud**, people removed: everything that accumulates scans over time (TARE coverage, SC-PGO keyframes).

The static cloud must never feed what avoids collisions. A detector mistake can then dirty a map, not cause a hit.
For the same reason the filter lets the clouds through unfiltered when the detector is late or down.

## Data flow

```
                     FAST-LIO
        /cloud_registered_body + /Odometry
                        │
                        ▼
            detector_cloud_adapter (B3)
                        │  /perception/detector/pointcloud   (sensor_level, PointXYZIRC)
                        ▼
            autoware_lidar_centerpoint (B4)
                        │  /perception/detector/objects      (sensor_level)
                        ▼
                people_tracker (B5)
              │                       │
     /perception/tracks      /perception/tracks_all          (camera_init)
              │                       │
              ▼                       ▼
   obstacle_adapter (B10)    dynamic_points_filter (B6)
              │                /cloud_registered      ─► /registered_scan_static        ─► TARE
              │                /cloud_registered_body ─► /cloud_registered_body_static  ─► SC-PGO
              │                                          /perception/dynamic_points     ─► rviz
              ▼
 /perception/dynamic_obstacles                               (camera_init)
              │
              ▼
 dynamic_obstacles_to_opti_pessi_obstacles.py                (scripts/)
              │
              ▼
 /opti_pessi/obstacles ─► OptiPessiController                (odom)
```

FAST-LIO stamps the two clouds, the odometry and the TF of a scan alike (the end of the scan). Every node here keeps
that stamp unchanged: it is the key that pairs clouds, detections and tracks.

## Running it

The launch file is included by `fast_lio_slam_launch.xml` of the `fast_lio` package, which forwards the arguments
below. **Do not launch `people_perception_launch.xml` by hand next to it**: a second filter would double every scan.

```bash
ros2 launch fast_lio fast_lio_slam_launch.xml people:=true
```

| Argument | Default | Effect |
|---|---|---|
| `people` | `false` | `false`: only `dynamic_points_filter`, as a pass-through. `true`: the whole chain, and the filter removes the tracked people |
| `centerpoint_model` | `~/RaNAV/models/lidar_centerpoint/v4.1/short_range` | Model folder of the detector |
| `opti_pessi_obstacles` | `true` | Relay the tracked people to `/opti_pessi/obstacles`. `false` when another node publishes that topic |
| `use_sim_time` | `true` | |

SC-PGO and TARE read their scans from `dynamic_points_filter` in both modes: **if the filter is not running, neither
gets any scan.** `people:=true` starts the detector, which needs a GPU with CUDA and TensorRT.

## Nodes

No parameter has a default in the code: each node reads everything from its file in `config/` and stops if one is
missing. The files carry the meaning and the unit of every value.

### `detector_cloud_adapter` (B3)

Turns the body cloud of FAST-LIO into what a CenterPoint node takes.

- **Layout**: `PointXYZIRC` of `autoware_point_types`, 16 bytes (`x y z` FLOAT32, `intensity` UINT8, `return_type`
  UINT8, `channel` UINT16). `return_type` and `channel` are 0.
- **Frame** `sensor_level`: origin and yaw of the sensor, no roll, no pitch. A pillar detector assumes a level floor
  at a constant height, and the body of a trotting quadruped rolls and pitches. The node publishes the TF
  `camera_init -> sensor_level` with every cloud; the tracker uses it to bring the detections back.
- **Crop**: the robot itself (`self_filter_radius`), `max_range_xy`, `z_min .. z_max`. A detector has to be trained on
  clouds cropped the same way.
- A cloud is paired with the odometry message of the same stamp. The small odometry message can arrive after the next
  cloud, so clouds wait for theirs in order (`cloud_queue`); a scan is dropped only when a later odometry is here and
  its own is not.

### Detector (B4): `autoware_lidar_centerpoint`

Not part of this repository: the CenterPoint node of Autoware Universe, started by the launch file. It answers every
cloud on `/perception/detector/pointcloud` with one `autoware_perception_msgs/DetectedObjects` on
`/perception/detector/objects`, same header. Its parameters are in `config/centerpoint_common.param.yaml` (Autoware's
file with `world_frame_id: camera_init` and a tighter NMS).

- Needs CUDA and TensorRT. In the RaNAV workspace: CUDA toolkit 13.0 and TensorRT 10.13 (`m3_install.sh` at the
  workspace root), `LD_LIBRARY_PATH` including `/usr/local/cuda-13.0/lib64`.
- `centerpoint_model` is a folder with the two `.onnx`, `ml_package.param.yaml` and
  `detection_class_remapper.param.yaml`. The TensorRT engines are built there on the first start (minutes) and are not
  portable between machines or TensorRT versions.
- The pretrained Autoware models were trained on road data: **a proof that the chain runs, not a people detector for a
  warehouse.**

Measured with the pretrained `v4.1/short_range` model on the two walking actors of the Gazebo warehouse, headless,
51 s:

| | |
|---|---|
| A track on an actor | 97 % of the time |
| Tracks on nobody | 2 % |
| Actor points removed | 99 % |
| Confirmed tracks started for 2 people | 8 |
| Inference | 45 ms median, 100 ms p95 |
| GPU memory, with the simulation | 2.3 GB |

### `people_tracker` (B5)

Follows the people from scan to scan: it smooths their position, estimates their velocity, and keeps them for a while
when the detector misses them.

- Takes the `PEDESTRIAN` detections only and brings them to `camera_init` with the TF at their stamp. That frame
  drifts slowly but never jumps; the frame SC-PGO corrects jumps at every loop closure and would break the tracks.
- One Kalman filter per person (position and velocity on the plane, constant velocity). Detections are matched to
  tracks by the smallest total distance (Hungarian algorithm), within `gate_distance` and `max_plausible_speed`.
- A track starts **tentative**, is **confirmed** after `min_hits` detections in `hits_window` scans with a mean score
  of `min_score_confirm`, **coasts** on its prediction while no detection matches, and is dropped after `max_age`.
- Publishes one message per message received, an empty one included: the filter reads it as "this scan is done".

| Topic | Content |
|---|---|
| `/perception/tracks` | Confirmed tracks: the obstacles |
| `/perception/tracks_all` | Confirmed, tentative and coasting: what the filter removes |
| `/perception/tracks_markers` | Box, speed, velocity arrow |

In a `TrackedObject`, `object_id` is left empty (the tracks carry no identity), `twist.linear` is the velocity in
`camera_init` (not in the frame of the object, as Autoware has it), and the x y blocks of the pose and twist
covariances are those of the Kalman filter.

### `dynamic_points_filter` (B6)

Stands between FAST-LIO and what accumulates its scans.

| Input | Output | Reader |
|---|---|---|
| `/cloud_registered` (`camera_init`) | `/registered_scan_static` | TARE; `far_static/terrainAnalysis(Ext)` -> `/terrain_map_static`, `/terrain_map_ext_static` -> far_planner (exploration: the saved graph; inspections: its routes, unless `people_free:=false`) |
| `/cloud_registered_body` (`rslidar_lio`) | `/cloud_registered_body_static` | SC-PGO |
| | `/perception/dynamic_points` | what was taken out of `/cloud_registered`, to look at |

- The outputs keep the fields, the `point_step`, the frame and the stamp of the inputs; the points that stay keep every
  byte.
- `enabled: false` (`people:=false`): every cloud goes out as it came in, at once.
- `enabled: true`: a cloud waits for the tracks of its scan. Around each track of `/perception/tracks_all` a vertical
  cylinder is cut out: the box plus margins, plus `position_sigma_factor` times the position uncertainty of the track,
  which grows while it coasts. The body cloud is tested through the odometry of its scan.
- A cloud whose tracks do not come within `max_wait` goes by `on_timeout`: `last_tracks` (the tracks of the closest
  scan, moved on with their velocity, if within `max_track_staleness`; unfiltered otherwise), `passthrough` or `drop`.
- **The clouds of a topic go out in scan order**, whatever each one waits for. SC-PGO pairs each cloud with the next
  odometry in its queue: one cloud out of turn or doubled shifts every later pair and ruins the map.
- **One filter only**: the node waits `duplicate_check_wait` after its start and stops if an output is already
  published.
- `/diagnostics`: clouds filtered, clouds that went without their tracks (detector late or down), clouds dropped,
  points removed, longest wait.

### `obstacle_adapter` (B10)

The confirmed tracks as what an Optimistic-Pessimistic MPC needs of each person.

- Of the people within `max_range` of the robot, the `max_obstacles` closest, closest first.
- `radius`: half the longer side of the box plus a margin, clamped. `max_speed` comes from the class, never from the
  estimated velocity: a person standing still can start walking.
- Stamp and frame are those of the tracks (end of the scan, `camera_init`). Positions are not moved on and radii not
  widened for the latency: the consumer adds `max_speed * (time of use - stamp)` to each radius.
- One `DynamicObstacleArray` on `/perception/dynamic_obstacles` per tracks message, an empty one included: a consumer
  that stops receiving it knows the perception has stopped.
- `/perception/dynamic_obstacles_markers`: two circles per obstacle, its radius and
  `radius + max_speed * marker_horizon`.

### Relay to the controller

`OptiPessiController` reads `/opti_pessi/obstacles` (`legged_controllers/ObstacleArray`, frame `odom`). The relay
`dynamic_obstacles_to_opti_pessi_obstacles.py` (`people_perception/scripts`) is started by this launch with
`people:=true` unless `opti_pessi_obstacles:=false`.

- It needs the TF `odom -> lio_map -> camera_init`, which `lio_odom_alignment.py` publishes (started by the TARE and
  FAR launch files). Until then it relays nothing.
- Only type and position reach the controller. A pedestrian becomes a `HUMAN`; radius and maximum speed are then the
  ones of `opti_pessi_interface/config/task.info`, not the ones in the message.
- Each message on `/opti_pessi/obstacles` replaces the previous one. **Two publishers erase each other**: do not run
  the camera detector `yolo26_rgbd_detect` next to it, or pass `opti_pessi_obstacles:=false`.

## Messages

`people_perception_msgs/DynamicObstacle`

| Field | |
|---|---|
| `uint8 label` | `autoware_perception_msgs/ObjectClassification` value |
| `geometry_msgs/Point position` | Centre, in the frame of the array |
| `float32 radius` | Circle enclosing the footprint, without any margin for motion [m] |
| `float32 max_speed` | Fixed by the class [m/s] |
| `bool is_stationary` | |
| `float32 confidence` | |

`people_perception_msgs/DynamicObstacleArray`: a `std_msgs/Header` (stamp of the scan, frame `camera_init`) and
`DynamicObstacle[] obstacles`.

## Build

```bash
colcon build --packages-select people_perception_msgs people_perception
```

| Dependency | Where it comes from |
|---|---|
| `rclcpp`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `visualization_msgs`, `tf2`, `tf2_ros`, `tf2_eigen`, `diagnostic_updater`, Eigen | ROS 2 Jazzy |
| `autoware_perception_msgs` | In the RaNAV workspace built from `src/autoware_msgs` (tag 1.13.0, that package only) |
| FAST-LIO | Run time: `/cloud_registered`, `/cloud_registered_body`, `/Odometry` with the same stamp per scan |
| `legged_controllers` | Run time, the relay to `/opti_pessi/obstacles` (`people:=true`, `opti_pessi_obstacles:=true`) |
| `autoware_lidar_centerpoint` | Run time, with `people:=true` |
| `python3-numpy`, `rclpy`, `rosbag2_py`, `rosidl_runtime_py` | The Python scripts |

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
python3 people_perception/test/dynamic_points_filter_check.py

# with people:=true: the same points in the same order but those taken out; /perception/dynamic_points is exactly what
# is missing; a detector cloud and tracks for every scan
python3 people_perception/test/dynamic_points_filter_check.py --people

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

- Pedestrians only. Other classes of a detector are ignored by the tracker and by the obstacle adapter.
- No model trained on warehouse data yet: with the pretrained CenterPoint a person is often lost and tracked anew
  (8 tracks for 2 people in 51 s).
- The tracks carry no identity: nothing tells that the person of one scan is the person of the next, beyond the
  track itself going on.
- The first SC-PGO keyframe is the first scan it gets, before any detector has answered: its people stay in the map.
- The controller side takes position and type only, and the lidar people and the camera detector cannot both publish
  `/opti_pessi/obstacles`.
- The margins of the filter, the tracker tuning and the obstacle radii are starting values, to be tuned on recorded
  data.
