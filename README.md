# Smart Trash
Our school (International School of Beijing) recently implemented a recycling system replaces the original one trashcan into multiple bins, so that recycling can be processed more efficiently. However, students have a tough time differentiating and identifying which bin their trash should go. Currently, audits have shown that students are only throwing their trash into the correct bin 24% of the time. So we developed a prototype "smart" trashcan that utilizes computer vision to help students sort trash.  
## Data
We are currently in the process of data collection. Our goal is to collect around 5,000 images of trash at our school, as there are lots of school specific trash. To supplement this we are also using a database from Mindy Yang and Gary Thung. Using various techniques of data augmentation we were able to generate a sufficient amount of data to train our model.

## Results
|          Config File          |   Model  | Optimizer | Dropout | Dense Layers | Epochs | Validation Accuracy |
|:-----------------------------:|:--------:|-----------|:-------:|:------------:|:------:|:-------------------:|
|  nets/config/recycle_vgg.pth  | VGG16-bn |    Adam   |   0.3   |       2      |   200  |        90.51%       |
|  nets/config/recycle_vgg1.pth | VGG16-bn |    Adam   |   0.2   |       2      |   200  |        88.87%       |
| nets/config/recycle_vgg19.pth | VGG19-bn |    Adam   |   0.3   |       2      |   200  |        81.42%       |
