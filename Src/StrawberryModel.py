import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
import transforms as T
import numpy as np
import torch.utils.data
import utils
import cv2
import torchvision.models.segmentation

import torch
import torchvision.models as models
torch.cuda.empty_cache()
import os

import re
from PIL import Image
batchSize=2
imageSize=[600,600]

class StrawberryDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        trainDir= os.path.join(root, '../Data/Images')
        images = [ f for f in os.listdir(trainDir) if os.path.isfile(os.path.join(trainDir,f)) ]
        images.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.imgs = images

        maskDir = os.path.join(root, '../Data/instance_segmentation')
        maskDirs = [ f for f in os.listdir(maskDir) if os.path.isfile(os.path.join(maskDir,f)) ]
        maskDirs.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.masks = maskDirs
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        batch_Imgs=[]
        batch_Data=[]# load images and masks
        #for i in range(batchSize):
        # load images and masks
        img = Image.open(os.path.join(self.root, '../Data/Images/',self.imgs[idx])).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(os.path.join(self.root, '../Data/instance_segmentation/',self.masks[idx]))#.convert("RGB")
        # convert the PIL Image into a numpy array
        #r = ripe
        #g = unripe
        #b = partially ripe
        #r,g,b = mask.split()
        '''        
        r = np.array(r)
        r = np.sort(r)
        r = np.unique(r)
        g = np.array(g)
        g = np.sort(g)
        g = np.unique(g)
        b = np.array(b)
        b = np.sort(b)
        b = np.unique(b)
        '''
        mask = np.array(mask)
        # first id is the background, so remove it
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #if num_objs==0: return loadData() # if image have no objects just load another image
        #boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        #for i in range(num_objs):
        #    x,y,w,h = cv2.boundingRect(masks[i])
        #    boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        #img = torch.as_tensor(img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   # there is only one class
        data["masks"] = masks
        data["image_id"] = image_id
        data["area"] = area
        data["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, data = self.transforms(img, data)
        return img, data

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def main():

    fileDir = os.path.dirname(__file__)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = StrawberryDataset(fileDir, get_transform(train=True))
    dataset_test = StrawberryDataset(fileDir, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    data_loader_test = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 7

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

    # pick one image from the test set
    img, _ = dataset_test[0]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
    print(prediction)
    Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).show()
    Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()).show()
    
    #torch.save(model.state_dict(), 'model_weights.pth')
    torch.save(model, 'model2.pth')

if __name__ == '__main__':
    main()