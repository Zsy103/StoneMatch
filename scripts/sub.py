#!/opt/conda/envs/loc/bin/python3
import sys
import os

# test_dir = os.path.dirname(os.path.abspath(__file__))
# print(test_dir)
# sys.path.append(test_dir)
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# from Template import *
import Template   
import StoneMatcher


def show_image(msg,stoneMatcher):
    # 将 ROS 消息转换为 OpenCV 图像
    bridge = CvBridge()
    targetImg = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    print(stoneMatcher.matchAll(targetImg))
  

if __name__ == "__main__":
    stoneMatcher = StoneMatcher.StoneMatcher('src/stoneMatch/templates/', debug=True, displayBest=True, displayAll=False)
    # 初始化 ROS 节点和订阅者
    rospy.init_node("image_subscriber", anonymous=True)
    image_sub = rospy.Subscriber("/navigation/left_camera/image_raw", Image, show_image,callback_args=stoneMatcher)
    
    # 循环等待消息
    rospy.spin()