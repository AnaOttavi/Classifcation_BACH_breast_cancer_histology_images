# Deep_Learning_Project

**Dataset Description**

BACH dataset. (2019). CC BY-NC-ND. Retrieved from https://github.com/bupt-ai-cz/BreastCancerCNN

Microscopy images are labeled as normal, benign, in situ carcinoma, or invasive carcinoma according to the predominant cancer type in each image. The annotation was performed by two medical experts and images where there was disagreement were discarded.

The dataset contains a total of 400 microscopy images, distributed as follows:

Normal: 100
Benign: 100
in situ carcinoma: 100
Invasive carcinoma: 100
Microscopy images are on .tiff format and have the following specifications:

Color model: R(ed)G(reen)B(lue)
Size: 2048 x 1536 pixels
Pixel scale: 0.42 µm x 0.42 µm
Memory space: 10-20 MB (approx.)
Type of label: image-wise

**Summary of Main Steps and Findings in Exploration**

For the exploration of the dataset the following steps were taken:
Visualization of random batches of images per class.
Confirmation of dataset balance.
Visualization of the average image per class and contrast of average Normal class image versus the average image of the three remaining classes.
The main findigs where:
The dataset only contains 400 images, this might represent a problem for the performance of the models, hence data augmentation will be explored in the models.
The dataset is balanced, containing the same number of samples per class.
The images file extention is .tif which is not supported by keras, therefore it will be converted to .png.
The color mode of the images is RGB.
The size of the images is quite large (2048 x 1536 pixels), therefore resizing will be required in the preprocessing for the sake of computational resources.
All images are of the same size and aspect ratio, thus no cropping will be needed.
Some patterns where found regarding pixel values of the images per class.

**Summary of Main Steps and Findings in Preprocessing**

For the preprocessing of the dataset the following steps where taken:
image resizing
one hot encoding of labels
pixel normalization
definition of image augmentation pipeline

**Summary of Main Steps and Findings in Handcrafted Models**

For the building of the handcrafted models the following steps where taken:

Two simple baseline models with one convolutional layer, one MaxPooling layer and one dense hidden layer where built, one with image augmentation and one without.
7 more complex models where built sequientially, each of them taking into account the performances and improvement needs of the previous one.
Adam was our choice of optimizer given that it is currently the most popular optimizer used for CNN models.
The starting learning rate selected was 0.001.
To assess the models we checked the categorical accuracy in trainning, validation and test sets.
The main findigs where:

Both baseline models underperform with accuracies of ~25% in trainning, validation and test sets, which is expected from a random baseline model.
The first complex model (Model 2) almost doubled the accuracy scores just by adding more complexity to its architecture:
we added two additional convolutional layers, each one of them with subsequent MaxPooling layers.
we increased the filter sizes from 3x3 to 5x5.
we increased the strides from 1 to 2 to prevent the number of trainable parameters to grow too much.
we added and aditional dense hidden layer.
we increased the number of neurons in both dense hidden layers.
Since the first complex model was still underfitting, we decided to try three different strategies hoping to improve this:
First we tried adding two more convolutional layers before the last MaxPooling layer (Model 3A). This new arquitecture lowered the accuracy scores again to the same level as the baseline models.
Secondly we tried increasing the number of filters (Model 3B) and achieved similar results from Model 2.
Third, we decreased the learning rate under the hypothesis that maybe the optimizer was skipping the optimal solution (Model 3C). With this model we achieved better accuracy scores both in trainning and validation and slightly better results in test set, but now the model was overfitting.
In order to reduce overffiting we tried three strategies as well:
Adding one dropout layer after the second dense hidden layer (Model 4A) and adding two dropout layers each one after each dense hidden layer (Model 4B). The first one achieved similar results in train and validation sets as Model 3C but improved the results in test set. The secon one, kept the ~20% difference between train and validation scores, but worsened them compared to Model 3C, although it slightly improved the accuracy in the test set.
Applying image augmentation (Model 4C). lowered the accuracy scores again to the same level as the baseline models.
Based on these results we decided to advance to hyperparamet tunning with model 4A. Without hypertuning this model achieves the following performance in the test set:
Test Categorical Accuracy: 0.550000011920929
Test AUROC: 0.7655208110809326
<img width="800" alt="Screen Shot 2023-07-13 at 00 24 20" src="https://github.com/AnaOttavi/Deep_Learning_Project/assets/86486485/f05fa242-95d7-4bb5-8e4c-fc6a9d9b363f">
<img width="425" alt="Screen Shot 2023-07-13 at 00 25 50" src="https://github.com/AnaOttavi/Deep_Learning_Project/assets/86486485/5cf79fdc-4bcf-4e17-9009-be8321ace253">
<img width="791" alt="Screen Shot 2023-07-13 at 00 26 06" src="https://github.com/AnaOttavi/Deep_Learning_Project/assets/86486485/c028f250-0d5a-4782-a4d9-e2504a34a749">
<img width="789" alt="Screen Shot 2023-07-13 at 00 28 03" src="https://github.com/AnaOttavi/Deep_Learning_Project/assets/86486485/a3e7ec2e-ee7d-4aeb-aea3-44931ece8f90">


**Summary of Main Steps and Findings in Hyperparameter Tunning**

For hyper parameter tunning we decided to use Bayesian Optimization given our large search space and limitations in computational power. Unlike Random Search and Hyperband that choose the hyperparameter combintions randomly, this tuner first chooses a few combinations randomly and based on their performance it chooses the next best possible sets of hyperparameters. This means it takes into account the history of the hyperparameters which were tried, giving us a higher chance of achieving optimal hyperparameters.

Source: https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f

Based on Model 4A, we decided to try these different values for the following parameters:

Number of filters: 32 to 256 in steps of 32
Filter size: 3x3, 5x5 and 11x11
Strides: 1 and 2
Padding: same and valid
Number of nodes in hidden layers: 64 to 512 in steps of 64
Activation functions: Relu and Sigmoid
Learning rate: 0.0001 to 0.01
Optimizer: Adam, RMSprop and SGD
Our objective with Bayesian Optimization was to achieve the highest possible validation accuracy. We did 100 trials with 50 epochs each.

The two best models obtained in the hypertunnig process where evaluated but none of them achieved better results than our best handcrafted model. This was an indicator that to achieve better results it was probably not enough to adjust the hyperparameters but the arquitecture of our models needed to be different. Transfer Learning should help us with this issue.

**Summary of Main Steps and Findings in Transfer Learning**

For Transfer Learning we implemented two state-of-the-art deep learning models for image classification: ResNet50 and VGG16. Based on these models we made some fine tunning and achieved our best results. Both models were trainned with Adam optimizer and a learning rate of 0.001.

The additions made to ResNet50 were: one additional hidden layer, batch normalization and one dropout layer. With this model we achieved the following results:

Train Accuracy: 1.0000
Val Accuracy: 0.7917
Test Accuracy: 0.6499
The additions made to VGG16 were: image augmentation in the preprocessing, one additional hidden layer with L2 regularization and one dropout layer. With this model we achieved the following results:

Train Accuracy: 0.8507
Val Accuracy: 0.6250
Test Accuracy: 0.7500
Even though the last model had the best performance in the test set, the learning curves were a bit erratic, hence it may require further investigation and optimization to achieve better and more stable results.
