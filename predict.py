from torch.utils.data import DataLoader
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import sys
from tensorboardX import SummaryWriter
import scipy.ndimage
import scipy.misc
import time
import math
import tables
import random
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
import torch.onnx



import cv2

from unet import UNet #code borrowed from https://github.com/jvanvugt/pytorch-unet

import torch.nn.functional as F
#import pydensecrf.densecrf as dcrf
from PIL import Image
#from v2.UNETnew import R2U_Net, UNetnew
# dataname = "nucleiSmall"
dataname = "seedling"

ignore_index = -100  #Unet has the possibility of masking out pixels in the output image, we can specify the index value here (though not used)
gpuid = 0

# -------------------------------------- unet params -------------------------------------------------------------------
#these parameters get fed directly into the UNET class, and more description of them can be discovered there
n_classes = 2        #number of classes in the data mask that we'll aim to predict
in_channels = 3      #input channel of the data, RGB = 3
padding = True       #should levels be padded
depth = 5            #depth of the network
wf = 3               #wf (int): number of filters in the first layer is 2**wf
up_mode = 'upconv'   #should we simply upsample the mask, or should we try and learn an interpolation
batch_norm = True    #should we use batch normalization between the layers

# -------------------------------------- training params ---------------------------------------------------------------
batch_size = 5
patch_size = 256
num_epochs = 30
edge_weight = False            #edges tend to be the most poorly segmented given how little area they occupy in the training set, this paramter boosts their values along the lines of the original UNET paper
phases = ["train", "val"]      #how many phases did we create databases for?
validation_phases = ["val"]    #when should we do valiation? note that validation is time consuming, so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch




imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
checkpoint = torch.load("epistroma_unet_best_model.pth")
model = UNet(n_class=n_classes, in_channels=in_channels, filterCountInFirstLayer=8, param_1=1, param_2=2, param_3=3, param_4=4, param_6=6, param_8=8, param_12=12, param_16=16, param_24=24).to(device)

model.load_state_dict(checkpoint["model_dict"])
model.eval()

img2 = Image.open("test_4.png")
img = loader(img2).float()
tensor_image = Variable(img, requires_grad=True).unsqueeze(0).cuda()


output = model(tensor_image.to(device))
output = output.detach().squeeze().cpu().numpy()
output = np.moveaxis(output, 0, -1)
fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].imshow(output[:, :, 1])
ax[1].imshow(np.argmax(output, axis=2))
ax[2].imshow(img2)
fig.savefig("result_test_4.png")












