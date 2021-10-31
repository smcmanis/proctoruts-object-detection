import cv2
from darkflow.net.build import TFNet

options = {
 'model': 'yolo/yolo.cfg',
 'load': 'yolo/yolov3.weights',
 'threshold': 0.3
}
tfnet = TFNet(options

# read the color image and covert to RGBimg = cv2.imread(‘sample_img\sample_dog.jpg’, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# use YOLO to predict the image
result = tfnet.return_predict(img)