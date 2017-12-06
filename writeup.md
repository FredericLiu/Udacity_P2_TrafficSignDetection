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
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

use matplotlib to visualize the signs randomly.
And visualized the total data distribution with matplob bar, under the hep of pandas pivot_table.

[image9]: ./writeup_image/statistic.png "dataset distribution"

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
 

#### 3. To train the model, I used AdamOptimizer, wigh learning = 0.001, epochs = 15, keep_prob = 0.5

#### 4. My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 94.6%
* test set accuracy of 92.9%

The initial iterative approach I choose is:
firstly shuffle the whole training dataset, then split them in mini batches with batchesize=128
for i in epochs:
	for each batch:
		training_opration
		...
	show accuracy

The initial problem is overfitting is significant. So I choose to use dropout on two full connect layers to avoid overfitting, it works. but the outcome still has overfitting. 

I also tried to tune batchsize to 64 or tune leaning rate to 0.002/0.005/0.01. Found decrease batches will make it even more slow to train the model, and bigger learning rate might cause underfitting.

So the hyper-parameters I use is just use the initial ones:learning rate= 0.001, batch_size = 128

I also tried to tune epochs to 15 or 20, but the increassing epochs doesn't help a lot on validation accuracy, althouth accuracy of training set could be very close to 100%.

As for the model, I choose Lenet-5 as described in prevous course, and didn't change the parameters inside the model, cuz currently I can't figure out the parameters will have what influence on the outcome. I will keep tring and find more reference on it.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. The five images are located in the folder cutome_images

I download these five images from another similar program https://github.com/kenshiro-o/CarND-Traffic-Sign-Classifier-Project, in order to have a comparison with his outcome.

These five signs are in different shape including triangle, rectancular and circule.Also with different lightening condition and background color.

These five sign pictures are with higher resulotion than initial training set, so have to resize them to (32,32,3) before get them into model.

The problem is: Even after resize, could they also be considered as the same distribution as the training/validation set? Even though they are recognized well during this experiment, more test with more images are still needed.



#### 2. Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (120km/h) | Speed limit (120km/h) 						| 
| No vehicles     		| No vehicles								|
| Priority road			| Priority road								|
| Road work	      		| Road work					 				|
| Vehicles over 3.5 metric tons prohibited		| Vehicles over 3.5 metric tons prohibited    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.9%

#### 3. top 5 softmax probabilities for each image along with the sign type of each probability.  Please refer to this chapter in my jupyter notebook 

Take the first image as an example, the model is relatively sure that this is a Speed limit (120km/h) sign (probability of 0.977), and the image does contain a Speed limit (120km/h) sign. 

For visualization of the signs, I take reference of the https://github.com/kenshiro-o/CarND-Traffic-Sign-Classifier-Project, used function show_image_list() in his program to show the new images

[image10]: ./writeup_image/top5_visualisation.png "top 5 possibilities distribution"

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. I visualized two CNN layers in the Jupyter notebook. From the outcome we could see the first CNN layer mainly recognized the border, but I can't see clearly what characteristics the second CNN layer detect.



