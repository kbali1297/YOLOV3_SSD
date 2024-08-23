import os
import numpy as np
from PIL import Image 
import shutil

folder_path = './kitti-main'

if not os.path.exists(f'{folder_path}/test.txt') \
    and not os.path.exists(f'{folder_path}/train.txt'):
    test_txt = open(f'{folder_path}/test.txt', 'w')

    train_txt = open(f'{folder_path}/train.txt', 'w')

    all_files= os.listdir(f'{folder_path}/images_processed')
    train_idxs = np.random.choice(np.arange(len(all_files)), int(0.9*len(all_files)), replace=False)

    for i, file in enumerate(all_files):
        if i in train_idxs:
            train_txt.write(f'{file}\n')
        else:
            test_txt.write(f'{file}\n')
    
    train_txt.close()
    test_txt.close()

train_txt = open(f'{folder_path}/train.txt', 'r')
test_txt = open(f'{folder_path}/test.txt', 'r')

if not os.path.exists(f'{folder_path}/images'): os.mkdir(f'{folder_path}/images')
if not os.path.exists(f'{folder_path}/labels'): os.mkdir(f'{folder_path}/labels')
if not os.path.exists(f'{folder_path}/images/train'): os.mkdir(f'{folder_path}/images/train')
if not os.path.exists(f'{folder_path}/images/test'): os.mkdir(f'{folder_path}/images/test')
if not os.path.exists(f'{folder_path}/labels/train'): os.mkdir(f'{folder_path}/labels/train')
if not os.path.exists(f'{folder_path}/labels/test'): os.mkdir(f'{folder_path}/labels/test')

path_dict = {'train':train_txt, 'test':test_txt}

for split in path_dict.keys():
    for img_name in path_dict[split].readlines():
        img_name = img_name[:-1]
        img = Image.open(f'{folder_path}/images_processed/{img_name}') ##removing \n from the end
        img.save(f'{folder_path}/images/{split}/{img_name}')

        label_name = f'{img_name[:-4]}.txt'
        shutil.copyfile(f'{folder_path}/labels_processed/{label_name}',
                        f'{folder_path}/labels/{split}/{label_name}')
    






