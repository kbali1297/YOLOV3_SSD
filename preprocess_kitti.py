import os
import numpy as np
from PIL import Image

def convert_coords(xmin, ymin, xmax, ymax):
    xc = (xmin+xmax)/2
    yc = (ymin+ymax)/2
    w = (xmax-xmin)
    h = (ymax-ymin)

    return xc,yc,w,h

if not os.path.exists('./kitti-main/images_processed'):
    os.mkdir('./kitti-main/images_processed')
if not os.path.exists('./kitti-main/labels_processed'):
    os.mkdir('./kitti-main/labels_processed')
    
class_names = {}
for label_f in os.listdir('./kitti-main/labels'):

    with open(f'./kitti-main/labels/{label_f}', 'r') as f:
        for line in f:
            class_name = line.split(' ')[0]
            if class_name!='DontCare' and class_name!='Misc':

                if class_name not in list(class_names.keys()):
                    class_names[class_name] = 1
                else:
                    class_names[class_name] +=1

class_to_idx = {}
idx_to_class = {}
with open(f'./kitti-main/class.names.orig', 'w') as f:
    for i, class_name in enumerate(class_names.keys()):
        f.write(f'{class_name} {class_names[class_name]}\n')
        class_to_idx[class_name] = i
        idx_to_class[i] = class_name
img_idx = 0


for image in os.listdir('./kitti-main/images'):
    img = np.array(Image.open(f'./kitti-main/images/{image}').convert('RGB'), dtype=np.uint8)
    label_f = f'{image[:-4]}.txt'

    print(f'image: {image}, label: {label_f}')

    ## Each image can be divided into 3 square images along the width
    h, w = img.shape[0], img.shape[1]

    img_1 = Image.fromarray(img[:,:h,:]) 
    img_2 = Image.fromarray(img[:,h:2*h,:]) 
    img_3 = Image.fromarray(img[:,2*h:3*h,:])
    
    ## Resize all images to 352X352
    img_1, img_2, img_3 = img_1.resize((352, 352)), img_2.resize((352, 352)), img_3.resize((352,352))

    f_idx_1, f_idx_2, f_idx_3 = img_idx, img_idx+1, img_idx+2
    img_1.save(f'./kitti-main/images_processed/{f_idx_1:06d}.png')
    img_2.save(f'./kitti-main/images_processed/{f_idx_2:06d}.png')
    img_3.save(f'./kitti-main/images_processed/{f_idx_3:06d}.png')
    

    f1 = open(f'./kitti-main/labels_processed/{f_idx_1:06d}.txt','w')
    f2 = open(f'./kitti-main/labels_processed/{f_idx_2:06d}.txt','w')
    f3 = open(f'./kitti-main/labels_processed/{f_idx_3:06d}.txt','w')

    with open(f'./kitti-main/labels/{label_f}', 'r') as f:
        for line in f:
            bbox_info = line.split(' ')
            class_name, (xmin, ymin, xmax, ymax) = bbox_info[0], bbox_info[4:8] 
            
            #print(f'class_name: {class_name}, xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}')
            if class_name!='DontCare' and class_name!='Misc':
                xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
                xc, yc, w_, h_ = convert_coords(xmin, ymin, xmax, ymax)
                if xmin<h:
                    if xmax<h:
                        f1.write(f'{class_to_idx[class_name]} {xc/h} {yc/h} {w_/h} {h_/h}\n')
                    #else:
                    #    xc1,xc2 = (xmin+h)/2, (h+xmax)/2
                    #    w1_,w2_ = (h-xmin), (xmax-h)
                    #    f1.write(f'{class_to_idx[class_name]} {xc1/h} {yc/h} {w1_/h} {h_/h}\n')
                    #    f2.write(f'{class_to_idx[class_name]} {(xc2-h)/h} {yc/h} {w2_/h} {h_/h}\n')
                elif xmin<2*h:
                    if xmax<2*h:
                        f2.write(f'{class_to_idx[class_name]} {(xc-h)/h} {yc/h} {w_/h} {h_/h}\n')
                    #else:
                    #    xc2,xc3 = (xmin+2*h)/2, (2*h+xmax)/2
                    #    w2_,w3_ = (2*h-xmin), (xmax-2*h)
                    #    f2.write(f'{class_to_idx[class_name]} {(xc2-h)/h} {yc/h} {w2_/h} {h_/h}\n')
                    #    f3.write(f'{class_to_idx[class_name]} {(xc3-2*h)/h} {yc/h} {w2_/h} {h_/h}\n')
                elif xmin<3*h:
                    if xmax<3*h:
                        f3.write(f'{class_to_idx[class_name]} {(xc-2*h)/h} {yc/h} {w_/h} {h_/h}\n')
                    #else:
                    #    xc3,w3_ = (xmin+3*h)/2, (3*h-xmin)
                    #    f3.write(f'{class_to_idx[class_name]} {(xc3-2*h)/h} {yc/h} {w3_/h} {h_/h}\n')
    
    f1.close()
    f2.close()
    f3.close()

    #Remove images with no objects
    label_file_names = [f'{f_idx_1:06d}', f'{f_idx_2:06d}', f'{f_idx_3:06d}']
    for label_name in label_file_names:
        label_path = f'./kitti-main/labels_processed/{label_name}.txt'
        image_path = f'./kitti-main/images_processed/{label_name}.png'
        if os.path.getsize(label_path)==0:
            os.remove(label_path)
            os.remove(image_path)

    img_idx += 3

class_names_and_counts = {}
for label_f in os.listdir(f'./kitti-main/labels_processed'):
    with open(f'./kitti-main/labels_processed/{label_f}', 'r') as f:
        for line in f:
            class_label = idx_to_class[int(line.split(' ')[0])]
            if class_label not in class_names_and_counts.keys():
                class_names_and_counts[class_label] = 1
            else:
                class_names_and_counts[class_label] += 1

class_names_txt = open('./kitti-main/class.names', 'w')

for (class_name, class_count) in class_names_and_counts.items():
    class_names_txt.write(f'{class_name} {class_count}\n')

class_names_txt.close()
