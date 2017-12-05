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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

use matplotlib to visualize the signs randomly.

### Design and Test a Model Architecture

#### 1. As a first step, I decided to convert the images to grayscale, and then normalized the image.

I tried for kinds of input data: original color images, grayscale images, normalized color images and normlized grayscale images.

The normalized grayscale images perform best accuracy through the algrithom.

I didn't generate any additional data, currently the performance could reach requirements, I will try to augment training data later.

#### 2. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Normalized Grayscale image   			| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| output 400, with dropout      				|
| Fully connected		| output 84, with dropout   					|
| Fully connected		| output 43, this is logits outputed by model   |
 

#### 3. To train the model, I used AdamOptimizer, wigh learning = 0.01, epochs = 15, keep_prob = 0.5

#### 4. My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 94.6%
* test set accuracy of 92.9%

I choose to use dropout on two full connect layers to avoid overfitting, it works. but the outcome still has overfitting. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. The five images are located in the folder cutome_images


#### 2. Here are the results of the prediction:
', '', '', '', ''

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (120km/h) | Speed limit (120km/h) 						| 
| No vehicles     		| No vehicles								|
| Priority road			| Priority road								|
| Road work	      		| Road work					 				|
| Vehicles over 3.5 metric tons prohibited		| Vehicles over 3.5 metric tons prohibited    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.9%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a Speed limit (120km/h) sign (probability of 0.977), and the image does contain a Speed limit (120km/h) sign. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. I visualized two CNN layers in the Jupyter notebook. From the outcome we could see the first CNN layer mainly recognized the border, but I can't see clearly what characteristics the second CNN layer detect.



