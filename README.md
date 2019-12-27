# Image based Virtual Try-on

[[Dataset Partition Label]](https://drive.google.com/open?id=1Jt9DykVUmUo5dzzwyi4C_1wmWgVYsFDl)  

[[Sample Try-on Video]](https://www.youtube.com/watch?v=h-QWM92VLA0)  




**Dataset Partition** We present a criterion to introduce the difficulty of try-on for a certain reference image.
## The specific key points we choose to evaluate the try-on difficulty
![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/criterion.png)

We use the pose map to calculate the difficulty level of try-on. The key motivation behind this is the more complex the occlusions and layouts are in the clothing area, the harder it will be. And the formula is given,
## The formula to compute the difficulty of try-onreference image

![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/formula.png)

where t is a certain key point, Mp' is the set of key point we take into consideration, and N is the size of the set. 

## Sample images from different difficulty level

![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/difficulty.png)

## Sample Try-on Results
  
![image](https://github.com/switchablenorms/DeepFashion_Try_On/blob/master/images/tryon.png)


## License
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

## Dataset
**VITON Dataset** This dataset is presented in [VITON](https://github.com/xthan/VITON), containing 19,000 image pairs, each of which includes a front-view woman image and a top clothing image. After removing the invalid image pairs, it yields 16,253 pairs, further splitting into a training set of 14,221 paris and a testing set of 2,032 pairs.
