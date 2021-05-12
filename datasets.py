'''
datasets.py

Copyright (c) 2021 Sony Group Corporation

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
'''

import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import os
from PIL import Image
import numpy as np
import cv2

from util import OutliersRemoval, fft_reconstruction, ColorCorrection, normalize_image, normalize_image2

sensor_size_x = 453
sensor_size_y = 721
crop_start_x = 97
crop_start_y = 79
crop_end_x = 1003
crop_end_y = 1521
scene_size_x = 3171
scene_size_y = 103

def load_dataset(input_dir, mask_filename, background_image_filename, redux, batch_size, shuffled=False, img_res=(3936, 128), is_real_scene=False):
    dataset = LenslessDataset(input_dir, mask_filename, background_image_filename, redux, img_res, is_real_scene)
    return DataLoader(dataset, batch_size=1, shuffle=shuffled)

class LenslessDataset(Dataset):

    def __init__(self, input_dir, mask_filename, background_image_filename, redux, img_res, is_real_scene):
        super(LenslessDataset, self).__init__()
        self.input_dir = input_dir
        self.mask = np.double(np.array(Image.open(mask_filename)))
        self.bg_sensor = np.double(np.array(Image.open(background_image_filename)))
        self.target = os.listdir(self.input_dir)
        self.img_res = img_res
        self.is_real_scene = is_real_scene

        if redux:
            self.target = self.target[:redux]
 
        self.imgs = self.target

    def __getitem__(self, index):
        target_path = os.path.join(self.input_dir, self.target[index])
        sensor = np.double(np.array(Image.open(target_path)))

        if not self.is_real_scene:
            # Load and subtract background sensor image
            sensor = sensor - self.bg_sensor
            sensor[sensor < 0] = 0        

        # Crop and average sensor image
        sub_sensor = cv2.resize(sensor[crop_start_x:crop_end_x, crop_start_y:crop_end_y, :], (sensor_size_y, sensor_size_x), interpolation=cv2.INTER_AREA)

        # Reconstruct each color channel
        result = np.zeros((scene_size_y, scene_size_x, 3))
        for ch in range(0, 3):
            result[:, :, ch] = np.fliplr(np.transpose(fft_reconstruction(sub_sensor[:, :, ch], self.mask, scene_size_x, scene_size_y)))

        # Post-process result and output:
        if self.is_real_scene:
            result = OutliersRemoval(result)
            result = normalize_image2(result)
        else:
            corrected_img = ColorCorrection(result, sensor[crop_start_x:crop_end_x, crop_start_y:crop_end_y, :])
            result = normalize_image(corrected_img)


        # OpenCV to Pillow
        reconst_img = Image.fromarray(result)

        reconst_img = reconst_img.resize(self.img_res)
        source = tvF.to_tensor(reconst_img)
        return source

    def __len__(self):
        return len(self.imgs)
