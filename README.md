# Covid-19 Detection from Lung X-rays

Table of Contents
-----------------
  
  * [Description](#description)
  * [Dataset](#the-dataset)
  * [Deep network architecture](#the-network)
  * [Pipeline](#pipeline)
  * [Results](#results)
  * [Instructions to run](#instructions-to-run)

# Description 
Covid-19 Detection from Lung X-rays 

Corona - COVID19 virus affects the respiratory system of healthy individual & Chest X-Ray is one of the important imaging methods to identify the corona virus.

With the Chest X - Ray dataset, developing a Deep Learning Model to classify the X-Rays of Healthy vs Pneumonia (Corona) affected patients.

This model also powers a web application to classify the Corona Virus(Pneumonia) X-rays.

Screenshots 

![web_app](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/screenshots/web_app.png)
![web_app_classify](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/screenshots/web_app_classify.png)

__Notebook 1__ [Covid-19 Detection from Lung X-rays.ipynb](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/Covid_19_Detection_from_Lung_X_rays.ipynb) - 
* Data exploration,
* Data preparation,
* Defining CNN Model,
* Training and saving model.

__Notebook 2__ [Covid-19 Detection from Lung X-rays Web App.ipynb](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/Covid_19_Detection_from_Lung_X_rays_Web_App.ipynb)-
* Creates web application using flask that takes an image as input and classifies it using the pre-trained model 

Both the notebooks are properly documented.

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
  * rescale the image 1./255, rotatate by 90 degrees, shift width and height by 0.15,flip horizontally, zoom by 0.5. 
* Resize all proposals to 224x224 pixels
* Perform a Forward pass through the network for all proposals
* Get output from softmax layer , if output is > 0.5 label as positive else negative

# Results
The model was run for 20 epochs maximum , batchsize = 16.
Callback was implemented to check if validation loss was < 0.2 to stop the training and prevent overfitting 

We recieved 
* loss: 0.1242 - accuracy: __95.80%__ on the training set 
* val_loss: 0.1534 - val_accuracy: __93.57%__ on the validation set

![model_acc_graph](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/screenshots/model_acc_graph.png)

![model_loss_graph](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/screenshots/model_loss_graph.png)

Since the test set is not labeled , we cannot get accuracy

This model detects 57/58 positive cases from the full train data and 77 from the test data 

# Instructions To Run
All the instructions are well documented in the notebooks.

Notebook 1 [Covid-19 Detection from Lung X-rays.ipynb](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/Covid_19_Detection_from_Lung_X_rays.ipynb) 

Notebook 2 [Covid-19 Detection from Lung X-rays Web App.ipynb](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/Covid_19_Detection_from_Lung_X_rays_Web_App.ipynb) 

Note: Make sure that the files are getting downloaded on colab from this repository, The model is approx 200mb and Github lfs allows limited bandwidth, in this case download the model from github and upload manually.
