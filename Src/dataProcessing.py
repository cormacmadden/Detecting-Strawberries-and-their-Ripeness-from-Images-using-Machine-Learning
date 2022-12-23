import os
import cv2
import numpy as np

def import_images(amt = 0):
    fileDir = os.path.dirname(__file__)
    imagesPath = os.path.join(fileDir, '../Data/Images')
    imageFiles = [ f for f in os.listdir(imagesPath) if os.path.isfile(os.path.join(imagesPath,f)) ]
    if amt == 0: 
        amt = len(imageFiles)
    images = np.empty(amt, dtype=object)
    for n in range(0, amt): 
        images[n] = cv2.imread(os.path.join(imagesPath,imageFiles[n]),cv2.IMREAD_UNCHANGED )
    return images

def import_instances(amt = 0):
    fileDir = os.path.dirname(__file__)
    ins_segPath = os.path.join(fileDir, '../Data/instance_segmentation')
    ins_segFiles = [ f for f in os.listdir(ins_segPath) if os.path.isfile(os.path.join(ins_segPath,f)) ]
    if amt == 0: 
        amt = len(ins_segFiles)
    instances = np.empty(amt, dtype=object)
    for n in range(0,amt):
        instances[n] = cv2.imread(os.path.join(ins_segPath,ins_segFiles[n]),cv2.IMREAD_UNCHANGED )
    return instances

def import_ripeness(amt = 0):
    fileDir = os.path.dirname(__file__)
    ripe_segPath = os.path.join(fileDir, '../Data/instance+ripeness_segmentation')
    ripe_segFiles = [ f for f in os.listdir(ripe_segPath) if os.path.isfile(os.path.join(ripe_segPath,f))]
    if amt == 0: 
        amt = len(ripe_segFiles)
    ripeness = np.empty(amt, dtype=object)
    for n in range(0,amt):
        ripeness[n] = cv2.imread(os.path.join(ripe_segPath,ripe_segFiles[n]),cv2.IMREAD_UNCHANGED)
    return ripeness

def import_boxes(amt = 0):
    fileDir = os.path.dirname(__file__)
    boudingBoxPath = os.path.join(fileDir, '../Data/bounding_box')
    boudingBoxFiles = [ f for f in os.listdir(boudingBoxPath) if os.path.isfile(os.path.join(boudingBoxPath,f)) ]
    if amt == 0: 
        amt = len(boudingBoxFiles)
    boudingBoxes = np.empty(amt, dtype=object)
    for n in range(0,amt):
        f = open(os.path.join(boudingBoxPath,boudingBoxFiles[n]), "r")
        boudingBoxes[n] = f.readlines()
        f.close()
    return boudingBoxes
