"""Image denoising using discrete wavelet transform method"""

import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt

# I. obtenir la décomposition de l’image à plusieurs résolutions

# Load image
in_img = cv2.imread('inputs/cameraman.jpg', cv2.IMREAD_GRAYSCALE)
print("cameraman.png shape \n :", in_img.shape)

# Compute the wavelet transform of the image
# returns 2D Discrete Wavelet Transform
wavelet_type = 'db1'  # wavelet type
nb_level = 1  # number of levels
decomp = pywt.wavedec2(
    in_img, wavelet_type, mode='periodization', level=nb_level)
(cA, (cH, cV, cD)) = decomp  # coefficients

print("decomp[0] : \n", decomp[0])
print("decomp[0] : \n", decomp[0].shape)


# Renormalization (only for display issues
# (to better visualize the contrast with the bare eye)
decomp[0] = decomp[0] / np.abs(decomp[0]).max()  # normalization
for detail_level in range(nb_level):
    decomp[detail_level + 1] = [d/np.abs(d).max() for d in decomp[detail_level + 1]]


# Concatenate all discrete wavelet transform coefficients 
# into a single n-d array
arr, slices = pywt.coeffs_to_array(decomp)
print("arr.shape : /n", arr.shape)
print("slices: /n ", slices)

# Display
fig = plt.figure()
ax = plt.axes()
plt.imshow(arr, cmap=plt.cm.gray)
plt.title("Different resolutions of the original image obtained by DWT")
plt.savefig("figures/original_decomp.png")


# II. Compute DWT coefficients if given image
wavelet_type = 'db1' # Choix du type d'ondelettes
(cA, (cH, cV, cD)) = pywt.dwt2(in_img, wavelet_type)


# III. Reconstruct image with inverse DWT
reconstruct_sig = pywt.idwt2((cA, (cH, cV, cD)), wavelet=wavelet_type)


# Q1
# Générer une version bruitée par un bruit Gaussien centré et d’écart type σ = 30: ce signal sera le signal observé y.
# Observer sa décomposition en ondelettes à plusieurs échelles: que remarque-t-on?

# a.
def addgaussian_noise(input_img, mean, sd, output_path=True):
    """Add guassian noise to image

    Args:
        input_img (np.array): image to add noise to
        mean (float): mean of the gaussian
        sd (float): sd of the gaussian
        output_path (str): path to save figure (noised image)

    Returns:
        noisyy_image: original image with gaussian noise.
    """

    gaussian = np.random.normal(mean, sd, input_img.shape)
    noisy_image = np.zeros(input_img.shape, np.float32)
    noisy_image = input_img + gaussian

    # save figure
    if output_path == True:
        plt.imshow(noisy_image, cmap=plt.cm.gray)
        plt.savefig(output_path)
    return noisy_image

# noised image
y = addgaussian_noise(in_img, mean=0, sd=30,
                      output_path="figures/y_noisy.png")


# b.
# Compute the DWT coefficients of signal y (= noised image)
y_decomp = pywt.wavedec2(y, wavelet_type, mode='periodization', level=nb_level)

# Renormalization (only for display issues
# (to better visualize the contrast with the bare eye)
y_decomp[0] = y_decomp[0] / np.abs(y_decomp[0]).max()  # normalization
for detail_level in range(nb_level):
    y_decomp[detail_level + 1] = [d/np.abs(d).max() for d in y_decomp[detail_level + 1]]

# Concatenating all discrete wavelet transform coefficients
y_arr, y_slices = pywt.coeffs_to_array(y_decomp)

# Display y DWT coeffecients on the same figure
fig = plt.figure()
ax = plt.axes()
plt.imshow(y_arr, cmap=plt.cm.gray)
plt.title("Forward discrete wavelet transform coefficients \n"
          + "of image with gaussian noise")
plt.savefig("figures/y_noisy_decomp.png")


# Q2.
# Ecrire un programme qui calcule un minimiseur x_hat de (1)
# pour l’opérateur L ci-dessus.


#a. y -(L)-> (cA, cV, CH, cD) (decomposition: DWT coefficients)

L = pywt.wavedec2(y, wavelet=wavelet_type, mode='periodization', level=nb_level)
(cA, (cH, cV, cD)) = L

print("print(cH.shape) :\n ", cH.shape)
print("print(cV.shape) :\n ", cV.shape)
print("print(cD.shape) :\n ", cD.shape)


#b. (cA, ncV, nCH, ncD) -(prox)-> (cA, ncV, ncH, ncD)

def proxf_vec (cX, lamb):
    """compute prox f of values in vector"""

    # check args
    assert len(cX.shape) == 1  # only vector

    ncX = np.zeros(cX.shape)
    for i, cX_i in enumerate(cX):
        if np.abs(cX_i) < lamb:
            ncX_i = 0
        elif cX_i >= lamb:
            ncX_i = cX_i - lamb
        elif cX_i < -lamb:
            ncX_i = cX_i + lamb
        ncX[i] = ncX_i
    return ncX


def proxf(cX, lamb):
    """compute prox f of values in matrix or vec"""
    # args check
    assert len(cX.shape) < 3  # cX must be only vector or matrix

    # if cX vec
    if len(cX.shape) == 1: 
        ncX = proxf_vec(cX, lamb)
    # if cX matrix
    elif len(cX.shape) == 2: # matrix
        ncX = np.zeros(cX.shape)
        for i in range(cX.shape[1]): # for each column
            tmp = cX[:, i]
            ncX[:, i] = proxf_vec(tmp, lamb)

    return ncX


(cA, (ncH, ncV, ncD)) = (cA, (proxf(cH, lamb=0.1),
                              proxf(cV, lamb=0.1),
                              proxf(cD, lamb=0.1)))

# c. (cA, ncV, ncH, ncD) -(L-1)-> x_hat (Reconstruction of image)
# inverse DWT
L_1 = pywt.idwt2((cA, (ncH, ncV, ncD)), wavelet=wavelet_type)


fig = plt.figure()
plt.imshow(L_1, cmap=plt.cm.gray)
plt.title("Image reconstruction (x_hat) using inverse DWT on gaussian noised image"
          + "\n according to our optimization problem")
plt.savefig("figures/y_noisy_reconstruct.png")


#3. Finding the best lambda that minimizes the mse between
# the original image (cameraman.png) and L_1 (after noising, optimizing and denoising)

def mse_img(img_original, img_denoised):
    """returns mse between two vectors of data"""
    return np.mean(np.square(img_original - img_denoised))


def best_lambda(original_img, lambdas, nb_level=1, mean=0, sd=30):

    # check args
    assert len(original_img.shape) == 2

    # initialization
    mse_results = np.zeros(len(lambdas))
    noised_img = addgaussian_noise(original_img, mean, sd, output_path=False)

    # compute mse for different lambdas
    for i, lamb in enumerate(lambdas):
        # a. decomposition
        L = pywt.wavedec2(noised_img, wavelet=wavelet_type,
                        mode='periodization', level=nb_level)
        (cA, (cH, cV, cD)) = L
    
        # b. 
        (cA, (ncH, ncV, ncD)) = (cA, (proxf(cH, lamb=lamb),
                                      proxf(cV, lamb=lamb),
                                      proxf(cD, lamb=lamb)))

        # c. reconstruction (x_hat)
        L_1 = pywt.idwt2((cA, (ncH, ncV, ncD)),
                         wavelet=wavelet_type)

        # mse original - decompressed
        mse_results[i] = mse_img(original_img, L_1)

    min_mse = np.min(mse_results)
    best_lambda = lambdas[np.argmin(mse_results)]

    return best_lambda, min_mse, mse_results


lambdas = np.linspace(0.0001, 200, 50)
best_lamb, min_mse, mse_vec = best_lambda(original_img=in_img,
                                          lambdas=lambdas)

print(" best lambda among {} is {} which returns mse={}".format(lambdas, best_lamb, min_mse))

fig = plt.figure()
plt.plot(lambdas, mse_vec)
plt.xlabel("lambda")
plt.ylabel("loss (mse)")
plt.scatter(x=best_lamb, y=min_mse, color="red",marker="o",
            label="best lambda = {}".format(round(best_lamb)), alpha=1)
plt.title("MSE depending on lambda (learning rate) value"
          + " \n between the original image and image x_hat")
plt.legend()
plt.savefig("figures/mse.png")
