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

    cv2.imwrite("result/img.jpg", img)
    cv2.imwrite("result/voc_img.png", voc_img)
    cv2.imwrite("result/predict_img.png", predict_img)