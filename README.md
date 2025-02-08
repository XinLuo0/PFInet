# Efficient Low-Light Light Field Enhancement With Progressive Feature Interaction
This repository contains official pytorch implementation of Efficient Low-Light Light Field Enhancement With Progressive Feature Interaction in TETCI 2024, by Xin Luo, Gaosheng Liu, Zhi Lu, Kun Li, and Jingyu Yang. 
CONTACT:luoxin_1895@tju.edu.cn

## Code
### Dependencies
* Pytorch 1.12.1
* CUDA 12.2
* Python 3.8
* Matlab(For data generation)
### Prepare Training and Test Data
* To generate the training data, please first download the L3F dataset and run:
  ```
  GenerateMatData.m
  GenerateDataForTraining.m
  ```
* To generate the test data, run:
  ```
  GenerateMatData.m
  GenerateDataForTest.m
  ```
### Train
* Run:
  ```
  python train.py
  ```
### Test
* Run:
  ```
  python test.py
  ```
