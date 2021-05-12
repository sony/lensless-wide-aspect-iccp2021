'''
util.py

Copyright (c) 2021 Sony Group Corporation

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
'''

import numpy as np

# FFT reconstruction
def fft_reconstruction(sub_sensor, mask, scene_size_x, scene_size_y):
    mask_fft = np.fft.fft2(mask)
    sensor_fft_conj = np.conj(np.fft.fft2(sub_sensor, mask.shape))
    result_large = np.fft.ifft2(mask_fft * sensor_fft_conj)
    return np.real(result_large[0:scene_size_x, 0:scene_size_y])


# Color correction:
def ColorCorrection(input_image_reconstruction, input_image_sensor):
    # Color correction parameters:
    R_RGB = np.array([[15320399.999999987, 425084.9999999994, 390341.24999999977], [1468736.2499999998, 22828492.49999998, 0], [0, 0, 32414516.249999963]])
    TOT_S_RGB = np.array([[137211868.0, 3807182.0, 3496127.0], [13155416.0, 204452721.0, 0], [0, 0, 290312556.0]])
    beta = 0.11100
    lower_outliers_num = 250
    upper_outliers_num = 20

    # Remove outliers from the data:
    for ch in range(0, 3):
        input_ch = input_image_reconstruction[:, :, ch].flatten()
        inds = np.argpartition(input_ch, lower_outliers_num)[:lower_outliers_num]
        input_ch[inds] = np.max(input_ch[inds])
        input_image_reconstruction[:, :, ch] = input_ch.reshape((input_image_reconstruction.shape[0], input_image_reconstruction.shape[1]))
        inds = np.argpartition(-input_ch, upper_outliers_num)[:upper_outliers_num]
        input_ch[inds] = np.min(input_ch[inds])
        input_image_reconstruction[:, :, ch] = input_ch.reshape((input_image_reconstruction.shape[0], input_image_reconstruction.shape[1]))
        approximate_offset_ch = np.sum(input_image_sensor[:, :, ch]) * beta
        if approximate_offset_ch < np.median(input_image_reconstruction[:, :, ch]):
            input_image_reconstruction[:, :, ch] = input_image_reconstruction[:, :, ch] - np.min(input_image_reconstruction[:, :, ch])
        else:
            input_image_reconstruction[:, :, ch] = np.zeros(input_image_reconstruction[:, :, ch].shape)

    # Color correction:
    alphas = (R_RGB - beta * TOT_S_RGB)
    alphas_inv = np.linalg.pinv(alphas)
    image_reconstruction_corrected = np.zeros(input_image_reconstruction.shape)
    for x in range(0, input_image_reconstruction.shape[0]):
        for y in range(0, input_image_reconstruction.shape[1]):
            image_reconstruction_corrected[x, y, :] = np.dot(alphas_inv, input_image_reconstruction[x, y, :])
    return image_reconstruction_corrected

# Image normalizer:
def normalize_image(data):
    data[data < 0] = 0
    data = data / (np.max(data))
    # Apply de-gamma (^(1.0/2.2)) // for TV images
    data = np.power(data, 1.0/2.2)
    p = np.uint8(np.round(255 * data))
    return p

# Image normalizer for real scenes:
def normalize_image2(data):
    for ch in range(0, 3):
        data_ch = data[:, :, ch]
        data_ch = data_ch - data_ch.min()
        data_ch = data_ch / data_ch.max()
        data[:, :, ch] = data_ch
    data = np.power(data, 1.0/2.2)
    p = np.uint8(np.round(255 * data))
    return p


# Remove outliers from the data:
def OutliersRemoval(data):
    lower_outliers_num = 250
    upper_outliers_num = 20
    for ch in range(0, 3):
        input_ch = data[:, :, ch].flatten()
        inds = np.argpartition(input_ch, lower_outliers_num)[:lower_outliers_num]
        input_ch[inds] = np.max(input_ch[inds])
        data[:, :, ch] = input_ch.reshape((data.shape[0], data.shape[1]))
        inds = np.argpartition(-input_ch, upper_outliers_num)[:upper_outliers_num]
        input_ch[inds] = np.min(input_ch[inds])
        data[:, :, ch] = input_ch.reshape((data.shape[0], data.shape[1]))
    return data