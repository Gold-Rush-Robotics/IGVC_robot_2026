#! /bin/bash

docker start dazzling_easley
docker exec -it dazzling_easley /bin/bash -c "
source /opt/ros/jazzy/setup.bash;
cd src/IGVC_robot_2026;
source install/setup.bash;
export DISPLAY=:0;
ros2 launch igvc_test_bringup sensor_launch.launch.py;
"