from network import RE_Net
import torch
import torch.nn as nn
import os
import h5py
from utils import event2tensor_time
import numpy as np
from skimage import color
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# get the SCER representation
def get_SCER_from_voxel(E):
    num_bin = E.shape[0]
    voxel = E[:, 0, :, :] - E[:, 1, :, :]
    re_voxel = np.zeros_like(voxel)
    left_voxel = voxel[:num_bin//2, :, :]
    right_voxel = voxel[num_bin//2:, :, :]
    right_voxel_sum = np.cumsum(right_voxel, axis=0)
    left_voxel = left_voxel[::-1]
    left_voxel_sum = np.cumsum(left_voxel, axis=0)
    left_voxel_sum = left_voxel_sum[::-1]
    re_voxel[:num_bin//2, :, :] = -left_voxel_sum
    re_voxel[num_bin//2:, :, :] = right_voxel_sum
    return re_voxel

def cal_res(blur_image,res_pre):
    output_image = blur_image - (torch.sum(res_pre,axis = 1,keepdim=True))/7
    return output_image

# data
data = h5py.File('Data/1-1-200-zju.h5', 'r')
blur_images = data['images']
events = data['events']
sharp_images = data['sharp_images']

# exposure
index = 1
start_time = blur_images['image{:09d}'.format(
    index)].attrs['exposure_start']
end_time = blur_images['image{:09d}'.format(
    index)].attrs['exposure_end']
img_size = blur_images['image{:09d}'.format(index)].attrs['size']
print(start_time,end_time,img_size)

# event-voxel
E, _ = event2tensor_time(events, img_size, start_time, end_time, num_bin = 6,voxel_type ='EFNet')
SCER = get_SCER_from_voxel(E)
blur_img = blur_images['image{:09d}'.format(index)]
sharp_img = sharp_images['image{:09d}'.format(index)]
SCER = torch.from_numpy(SCER)[None].cuda()[:,:,:-14].float()
blur_img = torch.from_numpy(color.rgb2gray(blur_img))[None,None].cuda()[:,:,:-14].float()
sharp_img = torch.from_numpy(color.rgb2gray(sharp_img))[None,None].cuda()[:,:,:-14].float()
print(SCER.shape,blur_img.shape,sharp_img.shape)

# network
net = RE_Net(blurry_channels=1,
            event_channels=6,
            out_channels = 6,
            rgb = False)
net = nn.DataParallel(net).cuda()
net.load_state_dict(torch.load('Pretrained_Model/RE_Net_GRAY.pth')['state_dict'])

# motion deblur
res_pre = net(blur_img,SCER)
deblur_img = cal_res(blur_img,res_pre)
print(deblur_img.shape)
blur_img = blur_img.detach().cpu().numpy()[0,0] * 255
deblur_img = deblur_img.detach().cpu().numpy()[0,0] * 255
sharp_img = sharp_img.detach().cpu().numpy()[0,0] * 255
cv2.imwrite(f'Result/blur_img.png',blur_img)
cv2.imwrite(f'Result/sharp_img.png',sharp_img)
cv2.imwrite(f'Result/deblur_img.png',deblur_img)
