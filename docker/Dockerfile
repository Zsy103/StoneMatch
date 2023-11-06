FROM nvidia/cuda:12.0.0-base-ubuntu20.04

# Install ROS Noetic
RUN apt-get update && \
    apt-get install -y curl gnupg2 lsb-release libffi7&& \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ros-noetic-desktop-full && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Create conda environment
ENV PATH="/opt/conda/bin:$PATH"
RUN /bin/bash -c "source ~/.bashrc && conda init && conda create -n loc python=3.10 -y"

# Activate conda environment
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate loc" >> ~/.bashrc 
RUN echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5" >> ~/.bashrc
ENV PATH /opt/conda/envs/loc/bin:$PATH

# Install Open3D and ROS support

RUN /bin/bash -c "source ~/.bashrc && pip install open3d==0.17.0 && \
    echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc && \
    echo 'source /root/catkin_ws/devel/setup.bash' >> ~/.bashrc"
# RUN /bin/bash -c "source ~/.bashrc && conda config --add channels open3d-admin && conda install -c open3d-admin open3d=0.17.0 && \
#     conda install -c open3d-admin open3d-ros=0.17.0 && \
#     echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc && \
#     echo 'source /root/catkin_ws/devel/setup.bash' >> ~/.bashrc"

# Set up catkin workspace
RUN /bin/bash -c "source ~/.bashrc && conda install -c conda-forge opencv && conda install libffi==3.3 && pip3 install empy catkin_pkg opencv-python pandas rospkg&& mkdir -p /root/catkin_ws/src && \
    cd /root/catkin_ws/src && \
    /bin/bash -c 'source /opt/ros/noetic/setup.bash && \
    catkin_init_workspace' && \
    cd /root/catkin_ws && \
    /bin/bash -c 'source /opt/ros/noetic/setup.bash && \
    catkin_make'"

# Set up entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# ENTRYPOINT ["/entrypoint.sh"]
