echo "please make sure running this script in root dir of stonematch. current working dir: $PWD"

docker run -it \
-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
-v "$PWD":/root/catkin_ws/src/stoneMatch \
--gpus all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e NVIDIA_VISIBLE_DEVICES=all \
stonematch:latest \
/bin/bash
