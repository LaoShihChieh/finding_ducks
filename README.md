![](https://i.imgur.com/hivrzTC.jpg)
# Pattern Recognition Assignment 1
###### `by 勞士杰 資工碩一 611021201       (discussed with 吳承翰)`
---
## Problem: Finding Ducks
---
Github: https://github.com/LaoShihChieh/finding_ducks
## Contents
**1. Problem description**
**2. Bayes Classifier**
**3. Gaussian Mixture Model**
**4. Setup**
**5. Process**
**6. Final Code**
**7. Experience**
**8. Reference**

---
## 1. Problem description
In this assignment, We are given an image of duck farm taken from a drone. We should use the Bayes classifier to extract the pixels of duck bodies from the image.
![](https://i.imgur.com/ojS6nIH.jpg width=35%)

## 2. Bayes Classifier
**Bayes’ Theorem** finds the probability of an event occurring given the probability of another event that has already occurred. Bayes’ theorem is stated mathematically as the following equation:

![](https://i.imgur.com/MXenGMd.png)
where A and B are events and P(B) ≠ 0.

**Posterior Probability:** the conditional probability of a class ***ω*** given an input feature vector, denoted as ***P(ω\x)***

**Likelihood:** the conditional probability that some feature vector ***x*** is observed in samples of a class ***ω***, donated as ***P(ω\x)***

**Class Prior Probability:** the probability of a class ***ω*** under the situation that no feature vector is given, donted as ***P(ω)***

**Predictor Prior Probability:** the probability that a feature vector ***x*** is observed, also called as ***Evidence***, donated as ***P(x)***.


* Basically, we are trying to find probability of event A, given the event B is true. Event B is also termed as evidence.
* **P(A)** is the **priori** of A (the prior probability, i.e. Probability of event before evidence is seen). The evidence is an attribute value of an unknown instance(here, it is event B).
* **P(A|B)** is a **posteriori probability** of **B**, i.e. probability of event after evidence is seen.

A **classifier** is a procedure by which the elements of the population set are each predicted to belong to one of the classes.

**Bayes Classifier:** Probabilistic model that makes the most probable prediction for new examples.

## 2. Gaussian Mixture Model
**Gaussian mixture models** are a probabilistic model for representing normally distributed subpopulations within an overall population.

**Mixture models** in general don't require knowing which subpopulation a data point belongs to, allowing the model to learn the subpopulations automatically. Since subpopulation assignment is not known, this constitutes a form of unsupervised learning.


**A Gaussian mixture model** is parameterized by two types of values, the mixture component weights and the component means and variances/covariances. A Gaussian mixture model with ***K*** components is stated mathematically as the following equation:



![](https://i.imgur.com/5CsNiwQ.png)

$\boldsymbol{\vec u_k}$***:*** the mean for the multivariate case.

$\boldsymbol{\sum _k}$***:*** the covariance matrix for the multivariate case.
## 3. Setup
#### **Hardware** :
* MacBook Air (Retina, 13-inch, 2020)
* Processor: 1.1 GHz Quad-Core Intel Core i5
* Memory: 8 GB 3733 MHz LPDDR4X
* Graphics: Intel Iris Plus Graphics 1536 MB
#### **Software**
* Python Libary
    * Numpy 1.21.5
    * OpenCV 4.5.4
    * Matplotlib 3.2.2
    * PyQt 5.15.6
* Labelme 4.6.0

## 3. Process
First I installed labelme and PyQt from [Labelme Github](https://github.com/wkentaro/labelme) and [PyQt Github](https://github.com/pyqt) with pip in the terminal:
`conda create -n=labelme python=3.7` 
`conda install pyqt`
`conda activate labelme`
`pip install labelme`
![](https://i.imgur.com/sjDDOh4.png =55%x)

Then I start the labelme api to annotate the image
`labelme`
![](https://i.imgur.com/83xecl2.jpg)
Next enter the folder of labelme\examples\semantic_segmentation to put in the jpg and json file, and create a labels.txt file
![](https://i.imgur.com/TIhS2Wk.png =55%x)
![](https://i.imgur.com/5OMrhi8.png =55%x)
Then run in the terminal the code to convert to voc dataset.
```
$ python labelme2voc.py <data> <data_output> --labels <label.txt path>
```
* ***<data>*** Path of label data(json and jpeg)
* ***<dataoutput>*** Path of output data of conversion
* ***<labels.txt path>*** Path of label.txt that includes all of label attributes.

You will then find a folder with the voc dataset, and within it you will find a SegmentationClassPNG with PNG images that we will be using later.

![](https://i.imgur.com/mhXKt1m.png =35%x)
    
## 6. Final Code
The first python code is model.py which includes different functions that will be used and the Gaussian Mixture Model.
### Get the pixels of the two class label from the image.
```python=
def get_pixels(img):
    duck_pixels = []
    not_duck_pixels = []
    for x in range(0,img.shape[0]):
        for y in range(0, img.shape[1]):
            if(img[x][y] == [128,0,0]).all(): # Red is duck pixels
                duck_pixels.append([x,y])
            if(img[x][y] == [0,128,0]).all(): # Green is not_duck pixels
                not_duck_pixels.append([x,y])

    return duck_pixels,not_duck_pixels
```
### Compute the mean vector $\boldsymbol{\vec u_k}$ of the pixels within the area.
```python=
def mean(pixels_array,img):
    RGB = [0, 0, 0]
    for i in range(0,len(pixels_array)):
        pixels_RGB = img[pixels_array[i][0]][pixels_array[i][1]]
        RGB += pixels_RGB
    mean_pixels_RGB = RGB // len(pixels_array)
    mean_pixels_RGB = np.array([mean_pixels_RGB]).T
    
    return mean_pixels_RGB
```
### Compute the covariance matrix $\boldsymbol{\sum _k}$ from the pixels list, the mean pixels, and the image.
```python=
def sigma(x_RGB_pixels_list,mean_RGB,img):
    sigma = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
    for i in range(len(x_RGB_pixels_list)):
        x_RGB = img[x_RGB_pixels_list[i][0]][x_RGB_pixels_list[i][1]]
        x_RGB = np.array([x_RGB]).T

        sigma += (x_RGB - mean_RGB) * ((x_RGB - mean_RGB).T)
    sigma = sigma/(len(x_RGB_pixels_list)-1)

    return sigma
```
### Gaussian Mixture Model with multivariate using the feature, the mean pixel, and the covariance matrix.
```python=
def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    sigma = np.matrix(sigma)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(sigma)**(1/2)) )
        x_mu = (x - mu)
        part2 = np.exp((-0.5) * (x_mu.T.dot(np.linalg.inv(sigma.T))).dot(x_mu))
        return part1 * part2
    else:
        raise NameError("The dimensions of the input don't match")
```
The second python code is main.py reads the images into the functions and models.
```python=
import model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    duck_vocimg_url = 'label_to_voc_dataset/voc_dataset/SegmentationClassPNG/full_duck.png'  # png image from labelme
    duck_img_url = 'label_to_voc_dataset/voc_dataset/JPEGImages/full_duck.jpg'  # original image
    voc_img = cv2.imread(duck_vocimg_url)  # Read png file
    voc_img = cv2.cvtColor(voc_img, cv2.COLOR_BGR2RGB) # Convert color to RGB
    img = cv2.imread(duck_img_url)  # Read original file

    predict_img = np.zeros((img.shape[0], img.shape[1], 3),dtype='uint8') # Initialize a imgage as large as the original image
    duck_pixels,not_duck_pixels = model.get_pixels(voc_img) # get the pixels of the labels

    duck_pixels_RGB_mean = model.mean(duck_pixels,img) # compute the mean of the duck pixels within the area

    not_duck_pixels_RGB_mean = model.mean(not_duck_pixels,img) # compute the mean of the not_duck pixels within the area
   
    duck_pixels_RGB_sigma = model.sigma(duck_pixels,duck_pixels_RGB_mean,img) # compute the covariance matrix of the duck pixels within the area
   
    not_duck_pixels_RGB_sigma = model.sigma(not_duck_pixels,not_duck_pixels_RGB_mean,img) # compute the covariance matrix of the not_duck pixels within the area

    duck_pixels_RGB_sigma = np.diag(np.diag(duck_pixels_RGB_sigma)) # transform matrix diagonal matrix for duck pixels
    not_duck_pixels_RGB_sigma = np.diag(np.diag(not_duck_pixels_RGB_sigma)) # transform matrix diagonal matrix for not_duck pixels

    for i in range(0, img.shape[0]):
        print('Predicting : {:d}/{:d}'.format(i, img.shape[0]))
        for j in range(0, img.shape[1]):
            x_array = np.array([img[i][j]])
            x_array = x_array.T
            possibility_of_duck = model.norm_pdf_multivariate(x_array, duck_pixels_RGB_mean, duck_pixels_RGB_sigma)
            possibility_of_non_duck = model.norm_pdf_multivariate(x_array, not_duck_pixels_RGB_mean, not_duck_pixels_RGB_sigma)
            a = possibility_of_duck[0] * (1/2000000)
            b = possibility_of_non_duck[0] * (1999999/2000000)
            if possibility_of_duck[0] >= possibility_of_non_duck[0]:
                predict_img[i][j] = [255, 255, 255]
            else:
                predict_img[i][j] = [0, 0, 0]

    duck_vocimg_url = 'result/voc_img.png'  # png image from labelme
    duck_img_url = 'result/img.png'  # original image
    duck_predict_img_url = 'result/predict_img.png' # result
    voc_img = cv2.cvtColor(voc_img, cv2.COLOR_BGR2RGB)
    predict_img = cv2.imread(duck_img_url)  # read result

    cv2.imwrite("result/10/img.jpg", img)
    cv2.imwrite("result/10/voc_img.png", voc_img)
    cv2.imwrite("result/10/predict_img.png", predict_img)
    img_url = 'label_to_voc_dataset/voc_dataset/JPEGImages/full_duck.jpg'
    voc_img_url = "result/10/voc_img.png"
    predict_img_url = "result/10/predict_img.png"
    
    img = cv2.imread(img_url)
    voc_img = cv2.imread(voc_img_url)
    predict_img = cv2.imread(predict_img_url)
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(voc_img)
    plt.subplot(1, 3, 3)
    plt.imshow(predict_img)
    plt.show()
```
![](https://i.imgur.com/BE9EQ0c.png  =55%x)

### Output
    
We could see from the two pictures that the classifier can extract most of the ducks.
    
![](https://i.imgur.com/ojS6nIH.jpg =35%x) ![](https://i.imgur.com/rZhnayx.png =35%x)

## 7. Discussion 

The issues that I faced during this assignments includes **understanding the math** behind the Gaussian Mixture Model, and **writing mathematical equations into code**. This skill can be improved by understanding and practicing more on coding in **Python**.

Another issue is the **hardware limitations** of my laptop computer, which results in **spending several hours** to run the final code. I struggled learning how to use the Linux system of my desktop computer, so I went with coding on my laptop. As a result, I borrowed my labmate's desktop computer which runs on Windows several times to test run the code. In the future I will try to understand how to run programs on Linux system to avoid spending too much time running programs with just a laptop.
    
## 8. Summary
In this assignment, I learned how to implement the the Bayes’ Classifier with the Gaussian Mixture Model to find extract specific pixels from an image. 
    
I also went through the process of learning the theory and math behind the classifier and model, and write it into code. 
    
I believe what the professor has taught us is a very important skill that will be benificial to us when we go into the workforce. 

This assignment gave me a great sense of accomplishment, and I can't wait to show off the results to my friends and family. It also made me start to think how I can implement this skill to other areas, such as extracting pixels to identify medical abnormalties.
    
## 9. Reference
Chapter 2. Bayes Classifiers: http://134.208.3.118/~ccchiang/PR/Ch2/ch2.html
Naive Bayes Classifiers: https://www.geeksforgeeks.org/naive-bayes-classifiers/
Classification rule: https://en.wikipedia.org/wiki/Classification_rule
A Gentle Introduction to the Bayes Optimal Classifier: https://machinelearningmastery.com/bayes-optimal-classifier/
Gaussian Mixture Model: https://brilliant.org/wiki/gaussian-mixture-model/
MathJax basic tutorial and quick reference: https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference
Notations and styles: https://hackmd.io/BXcdKqg7SD2OQqacCNYe0w?both
圖形識別 - 利用高斯混合模型(Gaussian Mixture Model)找出鴨子:  https://hackmd.io/eL4sW-GDR9Soc68pSSa8-Q
