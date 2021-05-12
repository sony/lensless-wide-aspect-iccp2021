'''
run.py

Copyright (c) 2021 Sony Group Corporation

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
'''

import os
import torch

import numpy as np
from PIL import Image

from u2net.u2net import U2NET
from datasets import load_dataset

def save_output(image_name,predict,d_dir):
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np = predict_np.transpose(1,2,0)
    im = Image.fromarray((predict_np*255).astype(np.uint8)).convert('RGB')
    im.save(d_dir+image_name)

if __name__ == "__main__":

    ######################################################
    # Parameter setting
    ######################################################
    device = torch.device('cuda:0')
    output_dir = os.path.join(os.getcwd(), 'results' + os.sep)
    model_path = os.path.join(os.getcwd(), 'data1/pretrained_model/pretrained.pth')
    img_res = (3936, 128)
    is_real_scene = True   # False: for "data1", True: for "data2"

    ######################################################
    # Data loader
    ######################################################
    if not is_real_scene:
        data_dir = 'data1/input'
    else:
        data_dir = 'data2/input'

    test_loader = load_dataset(data_dir, 'masks/basic_mask1.png', 'data1/allblack.png', 0, 1, shuffled=False, img_res=img_res, is_real_scene=is_real_scene)

    ######################################################
    # Load model
    ######################################################
    net = U2NET(3,3)
    net.load_state_dict(torch.load(model_path, map_location=device))
    if torch.cuda.is_available():
        net.to(device)
    net.eval()

    with torch.no_grad():
        for i_test, source in enumerate(test_loader):

            img_name = test_loader.dataset.imgs[i_test]
            print("processing:",img_name)

            if torch.cuda.is_available():
                source = source.to(device)

            d1,d2,d3,d4,d5,d6,d7= net(source)

            pred = d1[0,:,:,:]

            # save results
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            save_output(img_name,pred,output_dir)

            del d1,d2,d3,d4,d5,d6,d7

