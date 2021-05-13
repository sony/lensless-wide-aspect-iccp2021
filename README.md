# Lensless Mismatched Aspect Ratio Imaging

## Introduction
This repository contains the source code of the reconstruction method proposed in "Lensless Mismatched Aspect Ratio Imaging" (ICCP 2021).

The source code is ready-to-run and includes a pre-trained U2-net model generated from the modified KITTI dataset (KITTI-concatenated dataset).

The code is written in Python, and was tested using the Python 3.7.6 interpreter. The repository also contains examples of raw sensor images which can be directly used as input for the reconstruction code.
## How to use
When cloning the repository, please make sure that you have installed Git LSF on your computer.

run:
```
python run.py
```
The results will be stored in the "result" folder.

## Requirements
A CUDA supporting GPU is required to run this code.
For other Python package requirements, please refer to requirements.txt.

## License
This software is released under the MIT License, see LICENSE.
