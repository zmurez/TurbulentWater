# TurbulentWater
Code for "Learning to See through Turbulent Water" WACV 2018

Data and pretrained models are available at http://cseweb.ucsd.edu/~viscomp/projects/WACV18Water/

## Instructions
- download train.zip, val.zip and test.zip from http://cseweb.ucsd.edu/~viscomp/projects/WACV18Water/
- unzip train.zip into DATAROOT/Water
- unzip val.zip into DATAROOT/Water (note these images correspond to the ImageNet test set)
- unzip test.zip into DATAROOT/Water_Real
- dowanload the origional ImageNet training and test sets to DATAROOT/ImageNet/

python main.py --dataroot DATAROOT

## Minor modifications from the paper
- added a reconstruction (L1) and perceptual loss (VGG) to the output of the WarpNet
- all networks are trained for 3 epochs with all the losses from the start, with a constant learning rate of .0002
- all hyper-parameter weights are set to 1.0 except the perceptual and adversarial losses of the final output which are set to 0.5 and 0.2 respectively
- replaced transposed convolutions with nearest neighbor upsampling
- replaced instance normalization (and final layer denormalization) with group normalization
- added a 7x7 conv layer (to project from RGB to features and vice versa) to the begining and end of the networks

## Results
<img src="results.jpg" width="900px"/>
