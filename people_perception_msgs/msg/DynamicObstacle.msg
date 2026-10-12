# One tracked person (or other dynamic obstacle) for a planner that keeps out of a disk around it and assumes it can
# move in any direction at up to max_speed (block B10 of docs/perception_pipeline.md).

uint8 label                      # autoware_perception_msgs/ObjectClassification value
geometry_msgs/Point position     # centre, in the frame of the enclosing DynamicObstacleArray; z is unused by a 2D planner
float32 radius                   # r_obs [m]: circle enclosing the footprint, without any margin for motion
float32 max_speed                # [m/s], fixed by the class: it does not follow the estimated velocity
bool is_stationary
float32 confidence
