#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from model import myModel
def image_callback(msg,model):
    """Callback function to process the image."""
    bridge = CvBridge()
    try:
        # Convert the ROS Image message to a CV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #print image shape
        output = model.predict(cv_image)
        print(output.sigmoid())
        # Display the image
        cv2.imshow("Image Subscriber", cv_image)
        cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr(f"Error converting ROS Image to OpenCV: {e}")

def main():
    model_path = 'model.pth'
    mymodel=myModel(model_path, 'cpu')
    rospy.init_node('image_subscriber', anonymous=True)
    
    # Subscribe to the image topic (replace '/camera/image' with your topic)
    image_topic = "/jhu_daVinci/left/image_raw"
    rospy.Subscriber(image_topic, Image, lambda msg: image_callback(msg, mymodel))

    # Keep the node running
    rospy.spin()

    # Cleanup
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
