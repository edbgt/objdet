# Object Detection Node

Contains a ROS2 node to detect the cuboid and determine its position. To run with debug output enabled:

    ros2 run objdet_node objdet_exe --ros-args --log-level debug

Relies on the camera node for a point cloud stream published to `/tof_point_cloud`.
