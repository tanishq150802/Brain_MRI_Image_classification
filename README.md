# Brain_MRI_Image_classification
### Brain MRI Image classification into 15 categories - 14 cancer classes and 1 normal using Pytorch

By: [Tanishq Selot](https://github.com/tanishq150802) , Publically available dataset for training various CNN models can be found [here](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c). 

Refer to ```MRI Image classification.ipynb``` for the training, testing and inference scripts. Framework - Pytorch

## Approach
* The dataset I worked with had 573 Brain MRI images belonging to 15 classes - 14 cancer classes and 1 normal class. It can also be considered as belonging to 44 classes if subclasses - T1, T1C+ and T2 are also considered. As this would lead to even fewer images per class, I moved forward with training a model to classify the images into 15 classes. Moreover, many of the research works in this space include binary classification (tumor vs no-tumor) and classifying into 4 classes (normal, meningioma, glioma, and pituitary). Pytorch is the framework used and colab’s free GPU was used for training.

* Considering the small size of dataset provided for training the model, I moved forward with either training the well-known architectures fully or just fine-tuning them (just training the final layer). For preprocessing, I cropped the images till the periphery of the brain tissue as suggested by Krishnapriya S, Karuna Y. Pre-trained deep learning models for brain MRI image classification. This was done to avoid any background noise. As a part of data augmentation, random rotation and random horizontal flipping was also done while training, to prevent the models from overfitting.

* Pytorch’s WeightedRandomSampler was used to oversample the data from size 573 to 1152. This was done based on the class weights i.e; the 1 / class counts. The same weights were also used in the weighted loss function to avoid the bias against under-represented classes. Gradient clipping was also used during training to avoid exploding gradients and stabilize convergence. Due to the high number of parameters in the state-of-the-art models, weight decay was also used as a form of regularization for large weights. Learning rate scheduling was also added.

I used 374 images from the publicly available dataset for testing purposes. In order to further substantiate the good results obtained by efficientNet B0, I further tested the model on an enlarged test set (now having 640 images). Following classification report (on 640 test images):
![image](https://github.com/tanishq150802/Brain_MRI_Image_classification/assets/81608921/de7d57ce-dd64-4b33-8308-fe8256e429dc)

## Results
Training metrics on 574 image and testing metrics on 374 images.
![image](https://github.com/tanishq150802/Brain_MRI_Image_classification/assets/81608921/ebc9e2fe-571a-43ba-a3e9-da0f601d21f1)

## Requirements
* python
* torch
* torchvision
* scikit-learn
* matplotlib
