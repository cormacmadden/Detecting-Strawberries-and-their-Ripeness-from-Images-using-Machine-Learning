import cv2
import os
import numpy as np
from dataProcessing import *

def run():

    cv2.destroyAllWindows()
    fileDir = os.path.dirname(__file__)

    num_images = 5
    images = import_images(num_images)
    instances = import_instances(num_images)
    boxes = import_boxes(num_images)
    ripeness = import_ripeness(num_images)
    
    for i in range(0,num_images):
        cv2.imshow("image", images[i])
        cv2.imshow("instance", instances[i])
        cv2.imshow("ripeness", ripeness[i])
        print(boxes)
        print("\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    edges = cv2.Canny(images[4],20,500)
    cv2.imshow("edges",edges)

    #convert img to grey
    #img_grey = cv2.cvtColor(edges,cv2.COLOR_BGR2GRAY)
    
    #set a thresh
    thresh = 100
    #get threshold image
    ret,thresh_img = cv2.threshold(edges, thresh, 255, cv2.THRESH_BINARY)
    
    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # or use cv2.CHAIN_APPROX_SIMPLE
    # M = cv2.moments(cnt)

    #create an empty image for contours
    img_contours = np.zeros(images[4].shape)

    for cnt in contours:
        #if cv2.contourArea(cnt) > 50:
        if cv2.arcLength(cnt,False) > 100:
            cv2.drawContours(img_contours, cnt, -1, (0,255,0), 1)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img_contours,(x,y),(x+w,y+h),(0,255,0),2)

    
    cv2.imshow("grey",img_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    run()


