# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: documentation/visualization.png "Visualization"
[image2]: documentation/visualise_all_classes.png "Visualize all classes"
[image3]: documentation/grayscale.png "Grayscaling"
[image4]: test_images/37.go-straight-or-left.jpg "Traffic Sign 1"
[image5]: test_images/22.bumpy-road.jpg "Traffic Sign 2"
[image6]: test_images/13.yield.jpg "Traffic Sign 3"
[image7]: test_images/1.speed-limit.jpg "Traffic Sign 4"
[image8]: test_images/36.go-straight-or-right.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/goke-epapa/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the class labels are distributed across the training, validation and test datasets.
This visualisation shows the need for normalisation due to the huge variation in the distribution of classes across the datasets.

![alt text][image1]

Here is another visualisation all classes in the image, this image shows the need to perform some image processing due to very dark, low contrast and low brightness images.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to reduce the number of data to be analysed and also reduce the size of the data.

I also normalized the image data because of the huge variation in the distribution of classes across the datasets by using `cv2.equalizeHist()`

And finally, I applied the contrast limited adaptive histogram equalization on images to improve the clarity of the image features read [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html)

Here is an example of an original image and an augmented gray-scaled image:

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model uses the LeNet consisted of the following layers:


| Layer         		    |     Description	        				        | 
|:-------------------------:|:-------------------------------------------------:| 
| Input         		    | 32x32x1 RGB image   					            | 
| Convolution 5x5     	    | 1x1 stride, valid padding, outputs 28x28x32       |
| RELU					    |											        |
| Max pooling	      	    | 2x2 stride, outputs 14x14x32                      |
| Convolution 5x5	        | 1x1 stride, valid padding, outputs 10x10x64       |
| RELU					    |												    |
| Max pooling	      	    | 2x2 stride  outputs 5x5x64 		                |
| Flatten	      	        | outputs 1600 				                        |
| Fully connected		    | input 1600, output 400        					|
| RELU					    |												    |
| Fully connected		    | input 400, output 100        					    |
| RELU					    |												    |
| Fully connected (logits)	| input 100, output 43        				        |
 
The code for the architecture can be found in the `LeNet()` method.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the follow hyper parameters:

EPOCHS = 20
BATCH_SIZE = 64
rate = 0.001

Optimiser: Adam Optimiser

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.954 
* test set accuracy of 0.943

If a well known architecture was chosen:
* What architecture was chosen?
The LeNet architecture was used because it is a CNN that has been proven to be able to solve problems in a similar domain of image classification.

I tweaked the convolution layer depth to produce a better result as described below
Convolution Layer 1 - Increased depth from 6 to 32

Convolution Layer 2 - Increased depth from 16 to 64

* Why did you believe it would be relevant to the traffic sign application?
I believed it will be relevant because it had been used to classify handwritten and machine-printed characters before.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on from street maps: https://mapstreetview.com/

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Go straight or left	| Go straight or left   						|
| Bumpy road	      	| Bumpy road    					 			|
| Yield       		    | Yield   							            | 
| Speed limit (30km/h)  | Speed limit (50km/h)                          |
| Go straight or right	| Go straight or right      					|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 80%. This is rather low compared to the accuracy on the test set of 93%, so there is an indication of over fitting.

1. Go straight or left

The model was able to predict with 100% confidence because the image looks quite similar to those from the test set, however it could be a sign of overfitting

2. Bumpy road

The model was able to predict with 100% confidence as well because the features of the image are quite similar to that of the test dataset, however it could be a sign of overfitting

3. Yield

The model was able to predict with  83% confidence, the second predicted class traffic signals is also correct because the image has some part of a traffic signal, no overfiting here the model behaves really good 

4. Speed limit (30km/h)

The model wrongly predicted Speed limit (50km/h) with an 80% confidence due to the similarity in the signs, it is also possible that the dataset has more Speed limit (50km/h) than Speed limit (30km/h). The good thing is that the model predicts Speed limit (30km/h) in second place, and other speed limits are in 4th and 5th place. 

5. Go straight or right

The model was able to predict with 100% confidence because the image looks quite similar to those from the test set, however it could be a sign of overfitting

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image (Go straight or left), the top five soft max probabilities were:

```
Go straight or left: 1.000 
Turn left ahead: 0.000 
Keep left: 0.000 
Ahead only: 0.000 
Dangerous curve to the right: 0.000 

```

For the second image (Bumpy road), the top five soft max probabilities were:
```
Bumpy road: 1.000 
Bicycles crossing: 0.000 
Children crossing: 0.000 
Slippery road: 0.000 
Beware of ice/snow: 0.000  
```

For the second image (Yield), the top five soft max probabilities were:

```
Yield: 0.838 
Traffic signals: 0.158 
Priority road: 0.004 
General caution: 0.000 
Speed limit (80km/h): 0.000 
```

For the fourth image (Speed limit (30km/h)), the top five soft max probabilities were:

```
Speed limit (50km/h): 0.807 
Speed limit (30km/h): 0.129 
Wild animals crossing: 0.040 
Speed limit (80km/h): 0.017 
Speed limit (60km/h): 0.002 
```

For the fifth image (Go straight or right), the top five soft max probabilities were:
```
Go straight or right: 1.000 
Dangerous curve to the right: 0.000 
Keep right: 0.000 
Road work: 0.000 
End of speed limit (80km/h): 0.000
```

## References
[Converting Images to Grayscale](https://www.science-emergence.com/Articles/How-to-convert-an-image-to-grayscale-using-python-/)
- https://markbroerkens.github.io/CarND-Traffic-Sign-Classifier-Project/