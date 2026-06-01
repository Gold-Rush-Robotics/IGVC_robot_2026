docker start dazzling_easley
docker exec -it dazzling_easley /bin/bash -c "
cd src/IGVC_robot_2026;
source install/setup.bash;
export PYTHONPATH=\$PYTHONPATH:/opt/venv/lib/python3.12/site-packages;
export YOLOPV2_WEIGHTS=/root/ros2_ws/src/IGVC_robot_2026/models/yolopv2.pt
ros2 launch igvc_test_bringup nav2_gps_waypoint.launch.py use_sim_time:=false launch_motors:=true ;
"