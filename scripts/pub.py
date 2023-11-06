#!/usr/bin/env python

import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def publish_images():
    # 获取图片文件夹路径
    image_folder = "src/stoneMatch/testImg"

    # 获取图片文件列表
    image_files = sorted(os.listdir(image_folder))
    num_images = len(image_files)

    # 初始化 ROS 节点和发布者
    rospy.init_node("image_publisher", anonymous=True)
    image_pub = rospy.Publisher("/navigation/left_camera/image_raw", Image, queue_size=10)
    bridge = CvBridge()

    # 发布图片
    rate = rospy.Rate(2)  # 每秒发布一张图片
    i = 0
    step = 1
    while not rospy.is_shutdown():
        # 读取图片
        img_path = os.path.join(image_folder, image_files[i])
        img = cv2.imread(img_path)
        print('pub:', img_path)
        # 转换为 ROS 消息并发布
        img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        image_pub.publish(img_msg)
        
        # 更新图片索引
        i = (i + step) 
        if i==num_images-2 or i==0:
            step = -step
        

        # 等待下一次发布
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_images()
    except rospy.ROSInterruptException:
        pass