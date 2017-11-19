# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

Overview
---
In this project, we will use what we have learned about deep neural networks and convolutional neural networks to classify traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, we will then try out your model on images of German traffic signs that we find on the web.

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/orig_samples.png "Visualization"
[image2]: ./examples/histograms.png "Dataset Histograms"
[image3]: ./examples/grayscale.png "Grayscale"
[image4]: ./examples/augmentation.png "Imaging Functions"
[image5]: ./examples/augmented_result.png "Augmented Data Sample"
[image6]: ./examples/traffic_signs.png "Downloaded Traffic Signs"
[image7]: ./examples/predictions.png "Network Predictions"
[image8]: ./examples/probability.png "Probabilities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

My Python project notebook is located here [project code](https://github.com/duvitech/CarND-Traffic-Sign-Classifier-Project.git/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

Using traffic sign images found in the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).  
I loaded training, validation and test images for training my neural network.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

After loading the datasets I provided visual exploration of the data in the form of 
displaying an original image from each class:

![alt text][image1]


I furthered my understanging of the dataset by adding distribution histograms
of each dataset depicting the number of images in each class.

![alt text][image2]

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because ...

Here is a visualization of a sample set of a traffic signs after grayscaling.

![alt text][image3]

As a last step, I normalized the image data because and created functions for augmenting and enhancing the images for better results.

![alt text][image4]

The difference between the original data set and the augmented data set is the following
    - normalization
    - translation
    - scaled
    - warped
    - brightness adjusted

![alt text][image5]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x48 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x96 	|
| RELU          		|          									    |
| Max pooling	      	| 2x2 stride,  outputs 5x5x96    				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 3x3x172 	|
| RELU          		|          									    |
| Max pooling	      	| 1x1 stride,  outputs 2x2x172    				|
| Flatten				| 688        									|
| Fully Connected		| 84           									|
| Fully Connected		| 43											|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a 128 image batch size and 22 epochs.  My sigma was set to 0.1 and learn rate was set to 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] 


The first and third images might be difficult to classify because it is very hard to determine what the sign is a image off due to poor quality of the original image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                         |     Prediction	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Bumpy Road     		        | Bumpy Road  									| 
| Road Work    			        | Road Work 									|
| Slippery road                 | Wild Animal Crossing					|
| 30 km/h                       | 30 km/h                   					|
| Yield					        | Yield											|
| Turn Right Ahead		        | Turn Right Ahead								|
| Keep Right      		        | Keep Right					 				|
| 100 km/h			            | 100 km/h      				    			|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 87.5%. This compares less than favorably to the accuracy on the test set of 96.1%

![alt text][image7] 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

![alt text][image8] 


