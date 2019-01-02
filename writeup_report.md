# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/centered_image.jpg "Centered Image"
[image2]: ./examples/flipped_image.jpg "Flipped Image"
[image3]: ./examples/cropped_image.jpg "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model architecture and strategy

I used the NVIDIA model architecture with input normalization layer and cropping layer (50 px - from top, 20 px - from bottom)
The architecture includes 3 convolutional layer with 5x5 filter sizes, 2x2 subsample and depth from 24 to 48. After that it's applied two 3x3 layers of 64 depth. After that it's used 100, 50, 10 fully connected layers.
 For activation function used ELU layers (model.py lines 77 - 88)
I used the powerful architecture and a lot of data and augmentation. The simple models couldn't help me.

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting I used many methods as using a lot of extra data, using the flipped images and reverse track paths.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, the starting learning rate was 0.001. Then when finetuning the existent model I used the learning rates 0.0001 and 0.00001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and flipped images, I didn't use recovering, it worked not quite well.

For details about how I created the training data, see the next section. 

#### 5. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

To augment the data set, I also flipped images. Example:

![alt text][image2]

The preprocess of the data also included cropping to make model focus on the road instead of environment for generalization:

![alt text][image3]


All data that I used included 40000 images. I then preprocessed this data by ...

I finally randomly shuffled the data set and put 0.2 of the data into a validation set. 

The ideal number of epochs was 3 (I used ModelCheckpoint to find the best model)
