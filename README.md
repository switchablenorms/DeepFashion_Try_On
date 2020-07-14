## Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content, CVPR'20.
Official code for CVPR 2020 paper 'Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content'.
We rearrange the VITON dataset for easy access.

[[Dataset Partition Label]](https://drive.google.com/open?id=1Jt9DykVUmUo5dzzwyi4C_1wmWgVYsFDl)  [[Sample Try-on Video]](https://www.youtube.com/watch?v=BbKBSfDBcxI) [[Checkpoints]](https://drive.google.com/file/d/1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx/view?usp=sharing) 

[[Dataset_Test]](https://drive.google.com/file/d/1tE7hcVFm8Td8kRh5iYRBSDFdvZIkbUIR/view?usp=sharing) [[Dataset_Train]](https://drive.google.com/file/d/1lHNujZIq6KVeGOOdwnOXVCSR5E7Kv6xv/view?usp=sharing)


[[Paper]](https://arxiv.org/abs/2003.05863)

## Inference
```bash
python test.py
```
**Dataset Partition** We present a criterion to introduce the difficulty of try-on for a certain reference image.
## The specific key points we choose to evaluate the try-on difficulty
![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/criterion.png)

We use the pose map to calculate the difficulty level of try-on. The key motivation behind this is the more complex the occlusions and layouts are in the clothing area, the harder it will be. And the formula is given,
## The formula to compute the difficulty of try-onreference image

![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/formula.png)

where t is a certain key point, Mp' is the set of key point we take into consideration, and N is the size of the set. 
## Segmentation Label
```bash
0 -> Background
1 -> Hair
4 -> Upclothes
5 -> Left-shoe 
6 -> Right-shoe
7 -> Noise
8 -> Pants
9 -> Left_leg
10 -> Right_leg
11 -> Left_arm
12 -> Face
13 -> Right_arm
```
## Sample images from different difficulty level

![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/difficulty.png)

## Sample Try-on Results
  
![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/tryon.png)

## Training Details
For better inference performance, model G and G2 should be trained with 200 epoches, while model G1 and U net should be trained with 20 epoches.

## License
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

## Citation
If you use our code or models in your research, please cite with:
```
@InProceedings{Yang_2020_CVPR,
author = {Yang, Han and Zhang, Ruimao and Guo, Xiaobao and Liu, Wei and Zuo, Wangmeng and Luo, Ping},
title = {Towards Photo-Realistic Virtual Try-On by Adaptively Generating-Preserving Image Content},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Dataset
**VITON Dataset** This dataset is presented in [VITON](https://github.com/xthan/VITON), containing 19,000 image pairs, each of which includes a front-view woman image and a top clothing image. After removing the invalid image pairs, it yields 16,253 pairs, further splitting into a training set of 14,221 paris and a testing set of 2,032 pairs.