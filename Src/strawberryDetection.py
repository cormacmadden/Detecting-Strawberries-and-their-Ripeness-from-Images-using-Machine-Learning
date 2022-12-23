import cv2
import os
import numpy as np
from dataProcessing import *

def run():

    cv2.destroyAllWindows()
    fileDir = os.path.dirname(__file__)

    num_images = 5
    images = import_images(num_images)
    instancs = import_instances(num_images)
    boxes = import_boxes(num_images)
    ripeness = import_ripeness(num_images)

    for i in range(0,num_images):
        cv2.imshow("image", images[i])
        cv2.imshow("instance", instancs[i])
        cv2.imshow("ripeness", ripeness[i])
        print(boxes)
        print("\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    run()



