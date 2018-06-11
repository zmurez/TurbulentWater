import numpy as np
import os
from datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
from itertools import izip

rawroot='/mnt/Data1/Water_Real'
outroot='./results'
outname='concat'

datasets = []

# input images
datasets.append( ImageFolder(rawroot, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256)]), return_path=True) )

# results images
for exp_name in ['warp_L1', 'warp_L1VGG', 'color_L1VGG', 'color_L1VGGAdv', 'both_L1VGGAdv']:
    datasets.append( ImageFolder(os.path.join(outroot,'%s_test'%exp_name), return_path=True) )

# concat and save each image
for i, imgs in enumerate(izip(*datasets)):
    name = imgs[0][-1]
    print '%d/%d %s'%(i, len(datasets[0]), name)

    if not os.path.exists(os.path.join(outroot, outname, os.path.dirname(name))):
        os.makedirs(os.path.join(outroot, outname, os.path.dirname(name)))

    im = Image.fromarray( np.hstack((np.asarray(img[0]) for img in imgs)) )
    im.save(os.path.join(outroot, outname, name))

# concat best examples into figure
imgs=[]
for name in ['Tank/262A4109.JPG','Wild/262A4895.JPG','Wild/262A4984.JPG']:
    imgs.append( Image.open(os.path.join(outroot, outname, name)) )
im = Image.fromarray( np.vstack((np.asarray(img) for img in imgs)) )
im.save(os.path.join(outroot, outname+'.jpg'))
