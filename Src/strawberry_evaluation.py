import StrawberryModel
from old_strawberry_methods import StawberryDataset
import torch
from PIL import Image
import os
import transforms as T
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import utils
from torchvision.utils import draw_segmentation_masks
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
    model = torch.load('model.pth')
    # put the model in evaluation mode
    model.eval()
    #print(model)
    fileDir = os.path.dirname(__file__)

    dataset_test = StrawberryModel.StrawberryDataset(fileDir, get_transform(train=False))
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    for i in range(0,5):
        image, train_labels = next(iter(data_loader_test))
        imagesqueezed = image[0].squeeze()
        with torch.no_grad():
            prediction = model([imagesqueezed.to(device)])[0]
        num_obj = len(prediction["boxes"])

        proba_threshold = 0.9
        masks = prediction['masks'] > proba_threshold
        num_strawberries = len(masks)
        print(num_strawberries)
        masks = masks.squeeze(1)
        #boolean_masks = [
        #    out['masks'][out['scores'] > score_threshold] > proba_threshold
        #    for out in output
        #]
        #grid = make_grid(images)
        #show(grid)
        #imshow(imagesqueezed)
        image2 = torch.as_tensor(image[0]*255, dtype=torch.uint8)
        show(draw_segmentation_masks(image2, masks, alpha=0.7))
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