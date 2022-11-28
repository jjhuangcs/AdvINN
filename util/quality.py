import os
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr, compare_mse
import numpy as np
import lpips
import torch
import torchvision.transforms as T
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.ToTensor(),
])

path = r'./recover'
mse=0
psnr =0
ssim =0
lpipsnum =0
count =0
l_2sum =0
l_infsum=0
loss_on_vgg = lpips.LPIPS(net='vgg')
for parent, dirname, filenames in os.walk(path):
    indexlist = []
    is_exit = False
    for filename in filenames:
        index1 = filename.find('result')
        index2 = filename.find('cover')
        if (index1 >= 0):
            img1path = parent + '\\' + filename
            img1 = Image.open(img1path)
            img1_t = transform(img1)
            lp1 = lpips.im2tensor(lpips.load_image(img1path))
            img1 = np.array(img1)
            indexlist.append(index1)
            is_exit = True
        if (index2 >= 0):
            img2path = parent + '\\' + filename
            img2 = Image.open(img2path)
            img2_t = transform(img2)
            lp2 = lpips.im2tensor(lpips.load_image(img2path))
            img2 = np.array(img2)
            indexlist.append(index2)
        if (len(indexlist) == 2):
            mse += compare_mse(img1, img2)
            psnr += compare_psnr(img1, img2)
            ssim += compare_ssim(img1, img2, multichannel=True)
            lpipsnum += loss_on_vgg(lp1, lp2)
            l_2, l_inf = l_cal(img1_t, img2_t)
            l_2sum += l_2
            l_infsum += l_inf
            indexlist = []
            count += 1
    if(not is_exit):
        print(parent)
print("mse:" + str(mse/count))
print("psnr"+str(psnr/count))
print("ssim:"+str(ssim/count))
print("lpips:"+str(lpipsnum/count))
print("l_2:"+str(l_2sum/count))
print("l_inf:"+str(l_infsum/count))
print(count)
