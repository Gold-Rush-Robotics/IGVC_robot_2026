#! /bin/bash

docker start dazzling_easley
docker exec -it dazzling_easley /bin/bash -c "
cd src/IGVC_robot_2026;
source install/setup.bash;
ros2 launch igvc_test_bringup igvc_fused_drive.launch.py;