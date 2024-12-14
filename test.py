#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
from model import myModel
from collections import deque

def image_callback(msg,model,state_queue):
    """Callback function to process the image."""
    bridge = CvBridge()
    try:
        # Convert the ROS Image message to a CV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #print image shape
        output = model.predict(cv_image)
        print((output.sigmoid() > 0.5).int())
        state_queue.popleft()
        if (output.sigmoid() > 0.5).int() == 1:
            state_queue.append(1)
        else:
            state_queue.append(0)
        # Display the image
        # cv2.imshow("Image Subscriber", cv_image)
        # cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr(f"Error converting ROS Image to OpenCV: {e}")

def main():
    model_path = 'model.pth'
    mymodel=myModel(model_path, 'cpu')
    rospy.init_node('image_subscriber', anonymous=True)
    pub = rospy.Publisher('peg_visibility', Bool, queue_size=1)
    state_queue = deque([0]*4)
    # rospy.init_node('peg_toggle', anonymous=True)
    
    # Subscribe to the image topic (replace '/camera/image' with your topic)
    image_topic = "/jhu_daVinci/left/image_raw"
    rospy.Subscriber(image_topic, Image, lambda msg: image_callback(msg, mymodel, state_queue))

    # Keep the node running
    # rospy.spin()

    # Cleanup
    # cv2.destroyAllWindows()
    rate=rospy.Rate(5)
    while not rospy.is_shutdown():
        # print(state_queue)
        # print(sum(state_queue))
        if sum(state_queue) >= 3:
            pub.publish(Bool(True))
            print('peg detected')
        else:
            pub.publish(Bool(False))
        rate.sleep()

if __name__ == '__main__':
    main()
