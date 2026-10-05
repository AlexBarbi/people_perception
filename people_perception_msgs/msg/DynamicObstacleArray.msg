# The confirmed tracks of one scan. Published for every scan, an empty array included: a consumer that stops receiving
# it knows the perception has stopped.
std_msgs/Header header           # stamp: when the scan was measured (its end); the consumer adds max_speed * (time of
                                 # use - stamp) to each radius. frame_id: the FAST-LIO odometry frame (camera_init)
DynamicObstacle[] obstacles
