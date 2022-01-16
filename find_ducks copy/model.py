import numpy as np

# Get the pixels of the two class label from the image.
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

# Compute the mean vector μk of the pixels within the area.
def mean(pixels_array,img):
    RGB = [0, 0, 0]
    for i in range(0,len(pixels_array)):
        pixels_RGB = img[pixels_array[i][0]][pixels_array[i][1]]
        RGB += pixels_RGB
    mean_pixels_RGB = RGB // len(pixels_array)
    mean_pixels_RGB = np.array([mean_pixels_RGB]).T
    
    return mean_pixels_RGB

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