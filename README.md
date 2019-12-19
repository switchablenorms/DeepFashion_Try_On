# Image based Virtual Try-on

[[Dataset]](https://drive.google.com/file/d/1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo/view)  

![image](https://github.com/HAN-YANG-HIT/Fashion-Editing/blob/master/images/front.jpg)

**VITON Dataset** This dataset is presented in [[VITON]](https://github.com/xthan/VITON), containing 19,000 image pairs, each of which includes a front-view woman image and a top clothing image. After removing the invalid image pairs, it yields 16,253 pairs, further splitting into a training set of 14,221 paris and a testing set of 2,032 pairs.

**Dataset Partition** We present a criterion to introduce the difficulty of try-on for a certain reference image.
![image](https://github.com/HAN-YANG-HIT/Fashion-Editing/blob/master/images/criterion.png)

We use the pose map to calculate the difficulty level of try-on. The key motivation behind this is the more complex the occlusions and layouts are in the clothing area, the harder it will be. And the formula is given,

![image](https://github.com/HAN-YANG-HIT/Fashion-Editing/blob/master/images/formula.png)

where t is a certain key point, Mp' is the set of key point we take into consideration, and N is the size of the set. 

## Sample images from different difficulty level.

![image](https://github.com/HAN-YANG-HIT/Fashion-Editing/blob/master/images/difficulty.png)

## Sample Try-on Results
  
![image](https://github.com/HAN-YANG-HIT/Fashion-Editing/blob/master/images/tryon.png)


## License
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

