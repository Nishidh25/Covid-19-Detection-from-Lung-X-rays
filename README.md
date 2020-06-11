# Covid-19 Detection from Lung X-rays

Table of Contents
-----------------

  * [Dataset](#the-dataset)
  * [Deep network architecture](#the-network)
  * [Results](#results)
    * [Screenshots](#screenshots)
  * [Instructions to run](#instructions-to-run)


# The dataset:
We use [CoronaHack -Chest X-Ray-Dataset](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset)
Contains collection Chest X Ray of Healthy vs Pneumonia (Corona) affected patients infected patients along with few other categories such as SARS (Severe Acute Respiratory Syndrome ) ,Streptococcus & ARDS (Acute Respiratory Distress Syndrome)

The Dataset contains 58 Covid-19 positive X-rays

![dataset_division](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/screenshots/dataset_division.png)

# The Network
Inspired by the famous AlexNet and the FastRCNN paper. The inputs are 224x224 images, which are passed through a total of four convolution layers, then flattened to 1D tensors for classification.

Model Structure

![model_plot](https://github.com/Nishidh25/Covid-19-Detection-from-Lung-X-rays/blob/master/screenshots/model_plot.png)

# Results
