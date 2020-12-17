# 249 Final Project Documentation

Our goal is to build a Lane detection related application. To fulfill this goal we read several papers and try on different models. In this project, we use both traditional approaches as well as deep-learning approaches (lane segmentation) to design our application. 

## Focused areas
Our project together provides a lane detection, segmentation and navigation application. The accuracy is above 90 percent and our application can give navigation suggestions which is by telling the steering angle to let the drivers know how to steer the car. All pretrained h5 models are included.  
Overall Design

### Our project includes three major parts. 
- Traditional lane detection algorithm Open CV. 
- Deep Learning lane segmentation algorithm
- Autonomous Lane Navigation in Deep Learning


## Traditional Lane Detection Application
The problem that we are trying to solve in this part is to take a simple car-driving video from youtube as input data and process it to detect the lane within which the vehicle is moving. The application output should find a representative line for both the left and right lane lines and render those lines back to the video.

### Key Technique
- OpenCV
We use OpenCV to process the input images to discover any lane lines and rendering out a representation of the lane.
- Numpy / Matplotlib
Since images are dense matrix data, NumPy and matplotlib will be used to do transformations and render the image data.

### Implement Detail
#### Cropping the region of Interest

We want a region of interest that fully contains the lane lines. One simple shape that can help us to achieve this goal is to draw a triangle that starts at the bottom left corner of the images and then proceed to the center of the image. Then, we crop each driving image into the shape of a triangle as the region of interest.
![1](https://user-images.githubusercontent.com/54567577/102494772-27bc3680-40b0-11eb-880b-e50fe51c9fac.png)

#### Convert to different color spaces

Since we are detecting the lane line only, we do not care about the colors of the pictures at all. We only care about the differences between their intensity values. We can convert the image into grayscale to make the edge detection process simpler.
    
![3](https://user-images.githubusercontent.com/54567577/102494890-59350200-40b0-11eb-90f2-e762dd548b4a.png)

#### Detecting Edges in the Cropped Image

We use OpenCV’s library to use Canny edge detection to copped our images with some reasonable thresholds. Thus, we can get simple grayscale images for later lane detection.
  
 ![2](https://user-images.githubusercontent.com/54567577/102494854-47535f00-40b0-11eb-8d44-98a623875879.png)

#### Generate Lane Lines from edge pixels
We will use Hough Transform to transform all of our edge pixels into a different mathematical form. Once this is done, each edge pixel in “Image-space” will become a line or curve in “Hough Space”. Then, we can simply solve for the intersection between lines in hough space, and transform the intersection point back into image space to obtain a line that intersects enough edge pixels. An openCV function can help us to do hough transform easily.

 ![2](https://lh6.googleusercontent.com/V8qla2b_0XnHPutPW_DXpiGzZuxDnoFIC4H5o43wrW_PiknD1gNMyXTaBkN0X6US5BjTjvC-20Rmwwzv3KBn016vlUq-xoAMnw0nl8c)


#### Repeat the step above to process every image in the videos.
Video before annotated:
https://drive.google.com/file/d/1z_gZtwV-nirm3bEpsMA5QajfpVnDbgPe/view?usp=sharing


Video after annotated: 
https://drive.google.com/file/d/1vNAmM1TeS2L4nzdm3OTFrMb5Iph7_A-f/view?usp=sharing


## Lane Segmentation
Our goal is to automatically detect the road or lane boundaries. We will use a machine learning method which is semantic segmentation for this part. We will classify all the pixels in the scene and fit into predefined road categories. We use VGG 16 classifier and implement the model introduced by Shelhamer, Darrell and Long and also follow the instructions by 
azzouz marouen. 

### Key Technique
- FCN networks
- VGG 16 classifier

### Implementation Detail
#### Trained the model on the Kitti dataset.

#### Set up the environment on Google colab and checkout GPU
Versions Keras version: 2.4.3
#### Load data which downloaded previously from Kitti and accessible through Google Drives
![2](https://lh3.googleusercontent.com/Rn-N87z6mIO9onPwm5zbS3y_fGfn2F8sDoLL8BotDb2EOOz1tYC0w7-kwtUgx_QsyaziFop88eZDhuMkFrSOJkzV2J1iVLTNtrnUPfjF4wIoV07R7vRLv6jzcAJMLSlbWHPXZeoK)

#### Build the FCN 32 model based on the paper by Shelhamer, Darrell and Long and instructions from azzouz marouen
Plot out the training and validation accuracy and loss
![2](https://lh6.googleusercontent.com/qsBpHL7CW-IQHqVwxoFEwhlwc6h2TKqcxuqA-hmQ86xyalcMu2aJ-L8q6dDQ5h7tNTcA2hB9tvNxlUCw2cbr3RohYcMDIMqnfYP46lmEmyynOT29owsTQ_2RbJw9cX54LvR6hAKq)

![2](https://lh6.googleusercontent.com/XgyxzGYerornO95g8IxIZHcHaPvmCU8k25X1nDjWn6YRjKDVC-k8cfJ67ha3Qo9pC9uXzpeHvlKGcSjdb4k8StQg1xuj4BxZdggSKWa8CUBbKPSZcx8sZpZhVpaobBbxLzyvSgi3)

![2](https://lh3.googleusercontent.com/69CT8_NWWQQO6U41hdaaDDT_tnQ21BwiKHD41u16nJ_yiRqaiTL7cjL5O5xgjzDnQSycBMoBY-N89FMGbhrQGC0js17UAOLBwCodMx1Uso1XNjqK_Bl8gHt3HPgA_hN36mRPoY_7)

Build FCN 8 and compare it with FCN 32. Finally combine the result to the original image and plot out to see the result We also tried to build a FCN 8 model but the result did not look good. So we decided to use the FCN 32 model.
![2](https://lh4.googleusercontent.com/RdwJZxNs7UePfTpkGCT039D89nBjuXdtnra9uwThORbIgd7Jck8GIgtCcOUqIvlH6nbFQTbiFqeDBUqwzeKwfw11NxkOqdKlawvrDZ8G)

## Lane Navigation in Deep Learning
	This application is based on Nvidia’s paper which trains a (CNN) to map raw pixels from a front-facing camera directly to steering commands. Out input will be the video images from DashCam and the output will be the steering angle of the car. This model would use the video image and predict the steering angle of the car.

### Key Technique
- Convolutional Neural Network
![2](https://lh6.googleusercontent.com/jRAR1itMz0gYjAFQ6nrjh2HByGT8JDEoIfwd_R25eQnJB-nXq9CHjW7cBPafnip7GZreieaeheePompCswQtJPBVSTiD3FEDKPC3KiOkdcrPcUEgHiDeoA5GxK5sSMOkYEyeoWXB)

The above image is from Nvidia’s paper. It contains only 30 layers. The input image is 66* 200 pixel image. First, the image would be normalized, then passed through 5 groups of convolutional layers. Finally, the image would pass through 4 fully connected neural layers and generate a single output, which is the steering angle of the car.

![2](https://lh6.googleusercontent.com/jRAR1itMz0gYjAFQ6nrjh2HByGT8JDEoIfwd_R25eQnJB-nXq9CHjW7cBPafnip7GZreieaeheePompCswQtJPBVSTiD3FEDKPC3KiOkdcrPcUEgHiDeoA5GxK5sSMOkYEyeoWXB)

The predicted angle is then compared with the desired angle to give the video image. The error would be fed back into the CNN training process in backpropagation. As the graph shows above, this process is repeated in a loop until the loss is low enough. In fact, this is a typical image recognition training process, except that the output is a numerical value instead of a type of classified object.
 
### Implementation Detail
#### 1. Data Acquisition
![2](https://lh5.googleusercontent.com/QYpkDQeaqWnQIiLkQxpJ35Ee-ECpIoLLN9QGKSwP3Rbas2JRuO9EiKlLl51Aftqds5Zv73oONnbKvriOe9d8XKw9MjSYSIWnxoPbo8K2fedd0ON9DyoEEJ0qokr96T9xKo3GnV23)

Save Previous lane result as a new dataset. 
A steering angle is added.
#### 2. Image Augmentation
![2](https://lh3.googleusercontent.com/fGnOvJDApBBmAGDRlKVqQzLeIsvWhk9zGVdQl0fI21osjLerfDC1jIx0izJ7SSfGh5T5Mk2MwgViF1Rmr3V8sWcNyjHmCkPfMLtu4r3-r9eHTKCnUKykRo9tk-84AXZcMdMbNOam)

Since we only have a few hundred images, to train a deep network, we need a lot more images. Instead of running our car, let's try to augment our data. There are a couple of ways to do that.

##### Zoom: Crop out a smaller image from the center
![2](https://lh4.googleusercontent.com/edio_vlAJvk2ejiaKDdUuH7Qh4cif8dTbE34w4emKW9869IMeMPUp_2Z2kmh7dqx3Gq56NOTJLUiKERYBGNNcuLF4jtmf67kCLY3v-uZLjR89CQm6KtZw8uKx4fMfY3s1xGV9ND_)
</br>
##### Pan: crop out a smaller image from the left or right side
</br>
![2](https://lh5.googleusercontent.com/pl8da1uiMwC37WL6XDVguLx6xbxwMyS-xPGdH5u5KUXPwFxTd0P3PnVxT4HJUhUc6Gt2BpmjcoZwE2Vm-y7XQhcIKrMSPAksUiijEjspHGMQ8pLueUKw1IJWmemvYAhlgfDQswnJ)

##### adjust the brightness of the image
![2](https://lh5.googleusercontent.com/pl8da1uiMwC37WL6XDVguLx6xbxwMyS-xPGdH5u5KUXPwFxTd0P3PnVxT4HJUhUc6Gt2BpmjcoZwE2Vm-y7XQhcIKrMSPAksUiijEjspHGMQ8pLueUKw1IJWmemvYAhlgfDQswnJ)
##### flip the image horizontally, i.e do a left to right flip and change the steering angle correspondingly
![2](https://lh5.googleusercontent.com/79X5Id_itwh5cWOshpNmqXot5vNdB_OdoB2VZ2xA-xlv5EiE5hnWUKz8A5gGDA_a04zsosimj1nzm4iH_SkQUrMunoX-2vPPDN8KDov4jM9qoAymHFXvFwLnVAbycfa_FxEJ2ZKr)

#### 3. Image Process

The Nvidia model accepts input image in 200* 66-pixel resolution. Thus, we need to change our image into a suitable color space and size. First, we would crop out the top haft of the image because it is not relevant to the steering angle. Secondly, we would change the image to YUV color space.
![2](https://lh4.googleusercontent.com/thzymaXsFSbyweK5eUIA7m1cdStv2Kb0_vZsOzPTAPNNBP1Mu6VyrtvLgd0ANnHBvoI3P5X7ZfYLId5b1K5tSyduDMrNZ48MvWR9w-AndJsqz0k64uxM-nPkDtGsibcQ860QxPpJ)

#### 4. Training

We print out the parameter list. It shows that it has 250,000 parameters.
![2](https://lh6.googleusercontent.com/oYhPRSGuQ7CRLcz1Iroq_gEauAf17iIsLQDgHncc7ITtWMgw1L2qrPqCBJQd0R_t03OdQ-U_4iTHSAXNB2dbW8tEQ5X0shyMYl0Dgw_N7rCJuT-BAzTK9GC7RlrkiWTIqQ-Tvtuq)

### Evaluation
![2](https://lh5.googleusercontent.com/5eP-uG2E4RzjBoI5WfRKDFOSOGSxygipHDSL_JrjsRu2rqy2tUv8n5pb-0WE04poBTjIwy6MSzD23yr0G1VgnnFXRFpK_6ldbd080gEpURdX_pEePkuGsaNix0Z2Xg2Y03O5NTfH)

It is good to see that training loss and validation loss declined rapidly together, and both of them stay low after epoch 6. It seems that there is no overfitting issue because validation loss stayed low with training loss.

![2](https://lh5.googleusercontent.com/jZyStIa1HjWx3XxFOZvTbxy8vKsRrcdoxf8A9ZMiDTC4WPz7BvKn85XfPa4JUQLB9jS7GOdQyT5w2ooTvCgmCBZAHIypdRxZjb4h1WCO)

Another metric that seems to perform well is the R^2 metric. As we can see in our model, we have an R^2 of 93% even with 800 images, which is primarily because we used image augmentation.
