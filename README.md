# Strawberry Counting and Ripeness detection
### Deadline for Team report: Friday, January 6th, 2023
### Deadline for Individual report: Monday, January 9th, 2023
In the folder below you will find the data that you will use for Project 2. It consists of 3000 images of strawberries and their corresponding segmentation maps.
## Data file details:
Bounding Box - each row contains:
[Ripeness Class ID (0 = unripe, 1 = partially ripe, 2 = fully ripe ]<br>
[Normalized x value for bounding box centre coordinate (left to right)]<br>
[Normalized y value for bounding box centre coordinate (top to bottom)]<br>
[Normalized x value for bounding box width]<br>
[Normalized y value for bounding box width]<br>
## Instance Segmentation:
Greyscale image using intensity as the instance ID
Instance+Ripeness Segmentation:
RGB image using intensity as the instance ID and red channel, green channel and blue channel as class ID for ripe, unripe and partially ripe.
Image
The original hi-res photograph.<br>
We are holding back a further 100 images that are not in this dataset, that we will use to test your solutions.
## Project specification:
1. Perform a review of relevant papers that tackle similar problems - at least 9 related papers should be reviewed;
2. Based on the review in 1, select at least 3 different approaches that you wish to implement;
3. Your tasks are to count the number of strawberries in each image, and to identify the ripeness class (ripe, unripe, partially ripe) of each strawberry in that image;
NOTE: You may take different approaches to the counting and the ripeness classification tasks
4. Implement (training, validation and testing) these 3 approaches;
NOTE: you may use opensource code, as long as everything is fully acknowledged and documented
5. Write a team report, structured as an 8-10 page CVPR paper, as follows:
Introduction;
Background (review of at least 9 papers);
Implementation (what methods did you choose and why, description of your implementations);
Results (results of your validation and training, performance of your methods, in terms of both accuracy and speed);
Limitations, Conclusions and Future Work.
6. A  3-4 page individual report, outlining your own contribution to the project and the relative contributions of your team-mates.

NOTE: For groups with only two members, the requirement is to review at least 6 related papers and to implement at least 2 approaches.<br>
The templates for the CVPR format can be found at https://cvpr2022.thecvf.com/author-guidelines. (Scroll down to the section on "Submission Guidelines".}<br>
50% of the marks available will be allocated for the group report and the performance test that we will do with the remaining 100 images.<br>
The remaining 50% will be allocated based on the individual reports. By default, this will be the same as for the group report and results.<br> However, if some members make a larger or smaller contribution, this will be reflected in the individual marks.