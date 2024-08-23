import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=352, multiscale=True, transform=None, num_classes=2):
        list_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), list_path)
        split = "train" if list_path.split("/")[-1][:-4] == "train" or list_path.split("/")[-1][:-4] == "train_slt" \
            else "test"
        dataset_path = os.path.dirname(list_path)
        with open(list_path, "r") as file:
            self.img_names = file.readlines()
            self.img_files = [os.path.join(dataset_path, "images", split, img_name) for img_name in self.img_names]

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        self.num_classes = num_classes

    def __getitem__(self, index):
        
        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #boxes = np.loadtxt(label_path).reshape(-1, 5)
                boxes = []
                label_file = open(label_path, 'r')
                for line in label_file:
                    label_vals = line.split(' ')
                    boxes.append([int(label_vals[0]), float(label_vals[1]), 
                                  float(label_vals[2]), float(label_vals[3]),
                                  float(label_vals[4])])
                boxes = np.array(boxes)
                #if self.num_classes == 1:
                #    boxes[:, 0] = 0
        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except:
                print(f"Could not apply transform.")
                return
        else:
            img = transforms.ToTensor()(img)
            bb_targets = torch.zeros((len(boxes), 6))
            bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        img = resize(image=img, size=self.img_size)
        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))
        
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        
        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)
