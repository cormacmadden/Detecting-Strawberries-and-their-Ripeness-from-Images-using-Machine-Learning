import os
import cv2
import numpy as np
import json
import re
from PIL import Image , ImageOps
fileDir = os.path.dirname(__file__)

class OldStrawberryDataset():
    
    def import_images(self,imagesPath, amt = 0):
        imageFiles = [ f for f in os.listdir(imagesPath) if os.path.isfile(os.path.join(imagesPath,f)) ]
        imageFiles.sort(key=lambda f: int(re.sub('\D', '', f)))
        if amt == 0: 
            amt = len(imageFiles)
        images = np.empty(amt, dtype=object)
        for n in range(0, amt): 
            #images[n] = cv2.imread(os.path.join(imagesPath,imageFiles[n]),cv2.IMREAD_UNCHANGED )
            images[n] = Image.open(os.path.join(imagesPath,imageFiles[n])).convert("RGB")
        return images

    def import_instances(self,amt = 0):
        ins_segPath = os.path.join(fileDir, '../Data/instance_segmentation')
        ins_segFiles = [ f for f in os.listdir(ins_segPath) if os.path.isfile(os.path.join(ins_segPath,f)) ]
        ins_segFiles.sort(key=lambda f: int(re.sub('\D', '', f)))
        if amt == 0: 
            amt = len(ins_segFiles)
        instances = np.empty(amt, dtype=object)
        for n in range(0,amt):
            instances[n] = cv2.imread(os.path.join(ins_segPath,ins_segFiles[n]),cv2.IMREAD_UNCHANGED )
        return instances

    def import_ripeness(self,amt = 0):
        ripe_segPath = os.path.join(fileDir, '../Data/instance+ripeness_segmentation')
        ripe_segFiles = [ f for f in os.listdir(ripe_segPath) if os.path.isfile(os.path.join(ripe_segPath,f))]
        ripe_segFiles.sort(key=lambda f: int(re.sub('\D', '', f)))
        if amt == 0: 
            amt = len(ripe_segFiles)
        ripeness = np.empty(amt, dtype=object)
        for n in range(0,amt):
            ripeness[n] = cv2.imread(os.path.join(ripe_segPath,ripe_segFiles[n]),cv2.IMREAD_UNCHANGED)
        return ripeness

    def import_boxes_txt(self,amt = 0):
        boudingBoxPath = os.path.join(fileDir, '../Data/bounding_box/txt/')
        boudingBoxFiles = [ f for f in os.listdir(boudingBoxPath) if os.path.isfile(os.path.join(boudingBoxPath,f)) ]
        boudingBoxFiles.sort(key=lambda f: int(re.sub('\D', '', f)))
        if amt == 0: 
            amt = len(boudingBoxFiles)
        boudingBoxes = np.empty(amt, dtype=object)
        for n in range(0,amt):
            f = open(os.path.join(boudingBoxPath,boudingBoxFiles[n]), "r")
            boxesText = f.read()
            boudingBoxes[n] = boxesText.split()
            f.close()
        return boudingBoxes

    def import_boxes_json(self,amt = 0):
        boudingBoxPath = os.path.join(fileDir, '../Data/bounding_box/JSON')
        boudingBoxFiles = [ f for f in os.listdir(boudingBoxPath) if os.path.isfile(os.path.join(boudingBoxPath,f)) ]
        boudingBoxFiles.sort(key=lambda f: int(re.sub('\D', '', f)))
        if amt == 0: 
            amt = len(boudingBoxFiles)
        boudingBoxes = np.empty(amt, dtype=object)
        for n in range(0,amt):
            f = open(os.path.join(boudingBoxPath,boudingBoxFiles[n]), "r")
            boudingBoxes[n] = json.load(f)
        return boudingBoxes

    def boxes_to_json(self):
        boudingBoxPath = os.path.join(fileDir, '../Data/bounding_box')
        boudingBoxFiles = [ f for f in os.listdir(boudingBoxPath) if os.path.isfile(os.path.join(boudingBoxPath,f)) ]
        boudingBoxFiles.sort(key=lambda f: int(re.sub('\D', '', f)))
        dict1 = {}
        for idx, boudingBoxFile in enumerate(boudingBoxFiles):
            # creating dictionary
            dict1 = {}
            with open(os.path.join(boudingBoxPath,boudingBoxFile), "r") as fh:
                for i, line in enumerate(fh):
                    # reads each line and trims of extra the spaces
                    # and gives only the valid words
                    description = line.split()
                    dict1[i] = description
            
            boudingBoxOutPath = os.path.join(fileDir, '../Data/bounding_box/JSON/')
            if not os.path.exists(boudingBoxOutPath):
                os.makedirs(boudingBoxOutPath)
            out_file = open(boudingBoxOutPath + str(idx) + ".json", "w")
            json.dump(dict1, out_file, indent = 4, sort_keys = False)
            out_file.close()
        return 

    def draw_boxes(self,image,boxes):
        imwidth = image.shape[1]
        imheight = image.shape[0]
        for box in boxes.items():
            x = float(box[1][1])
            y = float(box[1][2])
            w = float(box[1][3])
            h = float(box[1][4])
            pt1x = int(imwidth*x - ((imwidth*w)/2))
            pt1y = int(imheight*y - ((imheight*h)/2))
            pt2x = int(imwidth*x + ((imwidth*w)/2))
            pt2y = int(imheight*y + ((imheight*h)/2))

            if box[1][0] == 0:
                color = (0,255,0)
            else: color = (0,0,255)
            cv2.rectangle(image,(pt1x,pt1y),(pt2x,pt2y),color,2)
        return image

    #def load_mask(self):