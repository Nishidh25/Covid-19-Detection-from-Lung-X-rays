# Covid-19 Detection from Lung X-rays

Table of Contents
-----------------

  * [Dataset](#the-dataset)
  * [Deep network architecture](#the-network)
  * [Pipeline](#pipeline)
  * [Results](#results)
    * [Screenshots](#screenshots)
  * [Instructions to run](#instructions-to-run)


# The Dataset:
We use [CoronaHack -Chest X-Ray-Dataset](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset)
Contains collection Chest X Ray of Healthy vs Pneumonia (Corona) affected patients infected patients along with few other categories such as SARS (Severe Acute Respiratory Syndrome ) ,Streptococcus & ARDS (Acute Respiratory Distress Syndrome)

The Dataset contains 58 Covid-19 positive X-rays

![dataset_division](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/screenshots/dataset_division.png)

# The Network
Inspired by the famous AlexNet and the FastRCNN paper. The inputs are 224x224 images, which are passed through a total of four convolution layers, then flattened to 1D tensors for classification.

Model Structure [skip image](#results)

![model_plot](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/screenshots/model_plot.png)

# Pipeline
* Select the image from csv and feed it to ImageDataGenerator
* Augment data 
  * rescale the image 1./255, rotatate by 90 degrees, shify width and height by 0.15,flip horizontally, zoom by 0.5. 
* Resize all proposals to 224x224 pixels
* Perform a Forward pass through the network for all proposals
* Get output from softmax layer , if output is > 0.5 label as positive else negative

# Results
The model was run for 20 epochs maximum , batchsize = 16.
Callback was implemented to check if validation loss was < 0.2 to stop the training and prevent overfitting 

We recieved 
* loss: 0.1242 - accuracy: __95.80%__ on the training set 
* val_loss: 0.1534 - val_accuracy: __93.57%__ on the validation set 

Since the test set is not labeled , we cannot get accuracy

This model detects 57/58 positive cases from the full train data and 77 from the test data 
