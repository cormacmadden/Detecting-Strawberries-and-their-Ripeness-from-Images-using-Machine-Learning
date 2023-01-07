import StrawberryModel
from old_strawberry_methods import OldStawberryDataset as OSD
import torch
from PIL import Image
import os
import transforms as T
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import utils
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.io import read_image
from torchvision.io import decode_image
from pathlib import Path
plt.ion()   # interactive mode

def main():

    #strawberry = StrawberryModel.StrawberryDataset()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load('model ripeness.pth')
    # put the model in evaluation mode
    model.eval()
    #print(model)
    fileDir = os.path.dirname(__file__)
    carol_test_path = os.path.join(fileDir, '../Data/Test Images')
    osd = OSD()
    carol_test_images = osd.import_images(carol_test_path)
    t_image = carol_test_images[0]
    tfms = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)])
    transform = transforms.Compose([
            transforms.PILToTensor()
    ])
    dataset_test = StrawberryModel.StrawberryDataset(fileDir, get_transform(train=False))
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    
    for img in carol_test_images:
        '''        
        image, train_labels = next(iter(data_loader_test))
        imagesqueezed = image[0].squeeze()
        with torch.no_grad():
            prediction = model([imagesqueezed.to(device)])[0]'''
        img_tensor= tfms(img).to('cuda').unsqueeze(0)
        with torch.no_grad():
            prediction = model(img_tensor)[0]
        num_obj = len(prediction["boxes"])
        #print(train_labels[0]["image_id"])
        proba_threshold = 0.5
        masks = prediction['masks'] > proba_threshold
        masks_colors=prediction["labels"]
        #masks = prediction['masks']> proba_threshold
        masks = masks.squeeze()
        masks = torch.as_tensor(masks, dtype=torch.bool)
        boxes = prediction['boxes'][prediction["scores"] > proba_threshold]
        box_colors=prediction["labels"][prediction["scores"] > proba_threshold]
        num_strawberries = len(boxes)
        print(num_strawberries)
        #boxes = boxes.squeeze(1)
        box_colors=box_colors.squeeze().cpu().data.numpy()
        print(box_colors)
        #image2 = torch.as_tensor(img, dtype=torch.uint8)
        image2 = transform(img)
        print(prediction["scores"])
        red = (255,0,0)
        green = (0,255,0)
        blue = (0,0,255)
        box_colors_tup = list()
        for i in range(0,len(box_colors)):
            if box_colors[i] == 1: box_colors_tup.append(red) # ripe
            elif box_colors[i] == 2: box_colors_tup.append(green) # unripe
            elif box_colors[i] == 3: box_colors_tup.append(blue) # partially ripe
        mask_colors_tup = list() 
        for i in range(0,len(masks_colors)):
            if masks_colors[i] == 1: mask_colors_tup.append(red) # ripe
            elif masks_colors[i] == 2: mask_colors_tup.append(green) # unripe
            elif masks_colors[i] == 3: mask_colors_tup.append(blue) # partially ripe
        show(draw_segmentation_masks(image2, masks,colors=mask_colors_tup, alpha=0.5))
        show(draw_bounding_boxes(image2, boxes,colors = box_colors_tup, width=3))
    return

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show(block=True)    

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show(block=True)    
if __name__ == '__main__':
    main()