import os
import numpy as np
import struct
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def bin_to_np_pcd(binFileName):
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return np_pcd

# def txt_to_np_pcd(txtFileName):
#     pc = []
#     with open(txtFileName, 'r') as f:
#         for line in f:
            

def cart_to_homo(pc):
    pc = np.concatenate([pc, np.ones( shape=(len(pc),1))], axis=-1)
    return pc


def read_proj_matrices(calib_file):
    calib_key = calib_file[:-4]
    proj_mats = {}
    proj_mats_shape = {}
    proj_mats_shape['P2'], proj_mats_shape['R0_rect'], proj_mats_shape['Tr_velo_to_cam'] = (3,4), (3,3), (3,4)
    proj_mats['P2'] ,proj_mats['R0_rect'], proj_mats['Tr_velo_to_cam'] = {},{},{}
    with open(f'{calib_file}', 'r') as f:
        for line in f:
            try:
                key, val = line.split(':')
                proj_mats[key] = np.array([float(v) for v in val.split()]).reshape(proj_mats_shape[key])
            except: continue

    RT = proj_mats['Tr_velo_to_cam']
    RT = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    R0 = proj_mats['R0_rect']
    R0 = np.concatenate([R0, np.array([[0,0,0]])], axis=0)
    R0 = np.concatenate([R0, np.array([0,0,0,1])[:,None]], axis=1)
    proj_mats['Tr_velo_to_cam'], proj_mats['R0_rect'] = RT, R0
    
    return proj_mats

def project_pc_to_img(calib_file, pc, scale=1):
    proj_mats = read_proj_matrices(calib_file)

    Y_img = proj_mats['P2']@proj_mats['R0_rect']@proj_mats['Tr_velo_to_cam']@pc.T
    Y_img = Y_img.T
    Y_img = np.concatenate([Y_img[:,:2]/Y_img[:,-1:], Y_img[:,-1:]], axis=-1)
    
    ## if the image is processed i.e obtained by splitting the original kitti image horizontally
    ## its calib matrices are stored in the calib_processed folder
    ## These projected points need to be shifted horizontally to display over the new images
    calib_folder = calib_file.split('/')[1]
    if calib_folder == 'calib_processed':
        offset_cond = int(calib_file[-10:-4])%3
        proj_offset = offset_cond * 375
    else: proj_offset = 0
    Y_img[:,0] -= proj_offset

    Y_img[:,:2] = Y_img[:,:2] * scale
    return Y_img


def projected_pc_to_camera_fov(img_path, proj_points):
    img = Image.open(img_path)
    img = np.transpose(np.array(img), (2,0,1))
    c, h, w = img.shape
    proj_indxs = (proj_points[:,0] > 0) * (proj_points[:,0] <w) * (proj_points[:,1] > 0) * (proj_points[:,1] < h)
    proj_points = proj_points[proj_indxs, :]

    return proj_points

def plot_proj_pc_on_img(img_path, proj_points, img_size=352):
    img = Image.open(f'{img_path}')

    img = np.array(img)
    print(f'Img shape before cv2: {img.shape}')
    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for pt_idx in range(proj_points.shape[0]):
        x, y = int( (proj_points[pt_idx][0])), int(proj_points[pt_idx][1])    
        color = cmap[int(510/proj_points[pt_idx][2]), :]
        cv2.circle(img, (x,y), radius=2, color=color)
    plt.imshow(img)

    return img

def plot_proj_pc_on_img_(img, proj_points, img_size=352):
    img_ = img.copy()
    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] #* 255
    for pt_idx in range(proj_points.shape[0]):
        x, y = int( (proj_points[pt_idx][0])), int(proj_points[pt_idx][1])    
        color = cmap[int(510/proj_points[pt_idx][2]), :]
        cv2.circle(img_, (x,y), radius=1, color=color)

    return img_

## Function to filter if LiDAR points if they lie inside a box with x1,x2 : xmin,xmax (similarly for y)
# box_corners = (x1,y1,x2,y2)  
def proj_pcs_inside(box_corners, proj_pc):
    '''
    Function that returns the projected point cloud present inside a box specified
    Args:
    box_corners : tuple of float values x1, y1, x2, y2 representing box corners
    proj_pc: proj point cloud with u,v and depth d with shape (N,3)
    Returns: 
    projected point clouds present in the box specified
    '''
    x1,y1,x2,y2 = box_corners
    idxs = (proj_pc[:,0] > x1) * (proj_pc[:,0] < x2) * (proj_pc[:,1] > y1) * (proj_pc[:,1] < y2)
    
    return proj_pc[idxs, :], np.arange(len(proj_pc))[idxs]