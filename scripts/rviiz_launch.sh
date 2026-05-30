#! /bin/bash

docker start dazzling_easley
docker exec -it dazzling_easley /bin/bash -c "
export DISPLAY=:0;
source /opt/ros/jazzy/setup.bash;
rviz2;