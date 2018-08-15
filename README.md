# inpainting
a re-implementation for deepfill v2 with some modifications

## TO DO
Now training with Places2 dataset, I will provide the code if the result is good enough.

## Some modification
Add perceptual loss after stage 1 and stage 2 for higher quality.

8/15 update: perceptual loss works, but sn patch gan is enough for good quality. The result is cleaner with perceptual loss but may hurt semantics at the begining. Need more training. 

## Some results
![image](https://github.com/uxtl/inpainting/blob/master/results/1.png)
![image](https://github.com/uxtl/inpainting/blob/master/results/2.png)
![image](https://github.com/uxtl/inpainting/blob/master/results/3.png)
![image](https://github.com/uxtl/inpainting/blob/master/results/4.png)
