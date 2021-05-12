# Lensless Mismatched Aspect Ratio Imaging

## Introduction
This repository contains the source code of the reconstruction method proposed in "Lensless Mismatched Aspect Ratio Imaging" (ICCP 2021).

The source code is ready-to-run and includes a pre-trained U2-net model generated from the modified KITTI dataset (KITTI-concatenated dataset).

The code is written in Python, and was tested using the Python 3.7.6 interpreter. The repository also contains examples of raw sensor images which can be directly used as input for the reconstruction code.
## How to use
run:
```
python run.py
```
The results will be stored in the "result" folder.

If you want to see the results for real scene data (data2 folder), change the value of `is_real_scene` from False to True in run.py.
```python
    is_real_scene = True   # False: for "data1", True: for "data2"
```


## License
This software is released under the MIT License, see LICENSE.txt.