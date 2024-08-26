import os
import numpy as np
from PIL import Image
from LiDAR_fusion import *

os.makedirs('./kitti-main/pcs_processed', exist_ok=True)
os.makedirs('./kitti-main/calib_processed', exist_ok=True)


img_idx = 0

for image in os.listdir('./kitti-main/images_orig'):
    img = np.array(Image.open(f'./kitti-main/images_orig/{image}').convert('RGB'), dtype=np.uint8)
    pc_f = f'{image[:-4]}.bin'

    print(f'image: {image}, pc: {pc_f}')

    ## Each image can be divided into 3 square images along the width
    h, w = img.shape[0], img.shape[1]

    img_1 = Image.fromarray(img[:,:h,:]) 
    img_2 = Image.fromarray(img[:,h:2*h,:]) 
    img_3 = Image.fromarray(img[:,2*h:3*h,:])
    
    ## Resize all images to 352X352
    #img_1, img_2, img_3 = img_1.resize((352, 352)), img_2.resize((352, 352)), img_3.resize((352,352))

    f_idx_1, f_idx_2, f_idx_3 = img_idx, img_idx+1, img_idx+2
    
    f1 = open(f'./kitti-main/calib_processed/{f_idx_1:06d}.txt','w')
    f2 = open(f'./kitti-main/calib_processed/{f_idx_2:06d}.txt','w')
    f3 = open(f'./kitti-main/calib_processed/{f_idx_3:06d}.txt','w')

    np_pcd = bin_to_np_pcd(f'./kitti-main/velodyne_reduced_pc/{pc_f}')
    pc = cart_to_homo(np_pcd) #NX4
    proj_pc = project_pc_to_img(f'./kitti-main/calib/{pc_f[:-4]}.txt', pc)#NX3

    proj_pc_1, idxs_1 = proj_pcs_inside((0, 0, h, h), proj_pc)
    proj_pc_2, idxs_2 = proj_pcs_inside((h,0,2*h,h), proj_pc) 
    proj_pc_3, idxs_3 = proj_pcs_inside((2*h, 0, 3*h, h), proj_pc)

    print(f'Total number of points: {pc.shape[0]}')
    print(f'Split Points: {proj_pc_1.shape[0]}, {proj_pc_2.shape[0]}, {proj_pc_3.shape[0]}')

    img_idx += 3

    np.save(f'./kitti-main/projected_pcs/{f_idx_1:06d}.npy', np_pcd[idxs_1, :])
    np.save(f'./kitti-main/projected_pcs/{f_idx_2:06d}.npy', np_pcd[idxs_2, :])
    np.save(f'./kitti-main/projected_pcs/{f_idx_3:06d}.npy', np_pcd[idxs_3, :])    
    
    with open(f'./kitti-main/calib/{image[:-4]}.txt', 'r') as f:
        for line in f:
            f1.write(line)
            f2.write(line)
            f3.write(line)