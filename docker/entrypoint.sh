#!/bin/bash

source ~/.bashrc
conda activate loc
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

exec "$@"
