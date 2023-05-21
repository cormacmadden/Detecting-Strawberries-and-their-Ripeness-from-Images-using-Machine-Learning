# Detecting Strawberries and their Ripeness from Images using Machine Learning (Mask R-CNN). 🍓


## Training Validation and Test Data:
The dataset used for this project is available for download at: <br/>
https://drive.google.com/drive/folders/17MSYZFUf4wW66PDjhrJbN2y3JKXxlZHW?usp=sharing <br/>

![Training Data.](/Figures/Training%20Data%20Strawberry%20Dataset.jpg?raw=true)

RGB image using intensity as the instance ID and red channel, green channel and blue channel as class ID for ripe, unripe and partially ripe. <br/> 

<br/>
🟥 - Ripe <br/>
🟩 - Not Ripe <br/>
🟦 - Partially Ripe <br/>


Fine-tuning a pre-trained model such as Mask R-CNN seemed like an effective approach to detecting strawberries and their ripeness as it can use the segmentation training images to find the strawberries and classification information such as ripeness can then also be added to the model.
The process used to fine-tune the model was taken from the tutorials on [pytorch.org](https://pytorch.org/vision/main/auto_examples/plot_transforms_v2_e2e.html#sphx-glr-auto-examples-plot-transforms-v2-e2e-py).
Tensorflow's torchvision wants you to write the implementation for the Dataset and Dataloader class to handle the storing loading and transformation of the training and testing data.

The first step of the process was to configure \__init__() function of the Dataset class so that it knows where all the images and information is stored. 
The next step is to configure the \__getitem__() function so that the DataLoader Class can interpret it. 
The \__getitem__() function loads in the files from the specified directories one by one. I decided to start with just the instance_segmentation files first and then use the instance+ripeness_segmentation files once that was up and working. From the mask image I calculated the bounding boxes for each strawberry by finding the min and max of every integer value on both axes. I count the number of unique values in the mask image and saves that that as the number of strawberries. 
The function then converts the boxes, masks, image_id and box_area to tensors.
It performs transforms on the image to convert it from a PIL image to a tensor and if it is training data it randomly flips half of the images. This process happens every time an image from the Dataset is requested by the DataLoader in the training process.

The model is pre-trained on COCO, a popular object detection and segmentation dataset with over 80 object categories.
It uses Fast R-CNN Predictor to predict the bounding boxes and Mask R-CNN to predict the mask.

The model was trained on the full 3000 images for 6 epochs and took about 4 hours and 36 minutes to complete on a GTX970 GPU. It was saved to a .pth file which is the file-type used by PyTorch to store the model weights. This file is 172MB in size 

### Results and Observations
The model was evaluated on the test dataset for every epoch. The results from the final epoch are shown in Table X. The 5 test images provided on blackboard have also been tested and the results look good. The lack of segmentation or bounding box data for the test images unfortunately means the intersection over union (IoU) or the accuracy of the ripeness classification cannot be measured. However, from a qualitative perspective, the masks seems to impressively detect all the strawberries and draw accurate masks over them. The classification of ripe, unripe and partially ripe also appear as I would have expected.

Image 759 was the most impressive from the test data as it detected a tiny strawberry in the background not even 10x10 pixels, I hadn't noticed it was there until it was detected, so it has surpassed the ability of at least one human anyway. However, the mask result for this image does detect a false positive on the upper right hand side. I set the threshold for a positive detection at 0.5, and so with some more fine tuning on the test data perhaps a better threshold could be found. 

![Results ](/Figures/Results%20Strawberry%20Detection.jpg)
### Limitations, Conclusions and Future Work
I think one of the limitations of the models is that they might not transfer as well as expected to new images with new lighting and taken in different environments. Future work could involve training and testing the models on new strawberry datasets to test their capabilities and strengthen its performance.
In conclusion I think the two models are very effective in their objective of detecting and classifying strawberries and their ripeness.


