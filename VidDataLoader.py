import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import pandas as pd
import os
from xml.dom import minidom
from PIL import Image
import torchvision.transforms.functional as TF


def list_files(root_path):
    bb_list = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            bb_list.append(os.path.join(path, name))
    return bb_list


def parse_annotation(ann):
    xml = minidom.parse(ann)
    x_min, x_max, y_min, y_max = int(xml.getElementsByTagName('xmin')[0].firstChild.data), \
                                 int(xml.getElementsByTagName('xmax')[0].firstChild.data), \
                                 int(xml.getElementsByTagName('ymin')[0].firstChild.data), \
                                 int(xml.getElementsByTagName('ymax')[0].firstChild.data)
    return [x_min, x_max, y_min, y_max]


def crop(img_desc):
    img = Image.fromarray(io.imread(img_desc['img']).astype('uint8'))
    height, width = img_desc['y_max'] - img_desc['y_min'], img_desc['x_max'] - img_desc['x_min']
    img_cropped = TF.crop(img, img_desc['y_min'], img_desc['x_min'], height, width)
    return np.array(img_cropped)
    # plt.imshow(img_cropped)
    # plt.show()


class VidDataset(Dataset):
    """VID dataset."""

    def __init__(self, xml_annotations_dir, root_dir, transform=None):

        self.annotations = list_files(xml_annotations_dir)
        self.images = list_files(root_dir)
        self.box_frame = pd.DataFrame(columns=['img', 'x_min', 'x_max', 'y_min', 'y_max'])

        for img, ann in zip(self.images, self.annotations):
            df_row = [img] + parse_annotation(ann)
            self.box_frame = self.box_frame.append(pd.Series(df_row, index=['img', 'x_min', 'x_max', 'y_min', 'y_max']),
                                                   ignore_index=True)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.box_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = crop(self.box_frame.iloc[idx])
        crop_coord = np.array(self.box_frame.iloc[idx, 1:]).astype('float32')
        sample = {'image': image, 'crop_coord': crop_coord}

        if self.transform:
            sample = self.transform(sample)

        return sample
