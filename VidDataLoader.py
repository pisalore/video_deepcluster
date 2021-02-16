import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np
import pandas as pd
import os
from xml.dom import minidom


def list_files(root_path):
    bb_list = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            bb_list.append(os.path.join(path, name))
    return bb_list


def parse_annotation(ann):
    xml = minidom.parse(ann)
    x_min, x_max, y_min, y_max = xml.getElementsByTagName('xmin')[0].firstChild.data, \
                                 xml.getElementsByTagName('xmax')[0].firstChild.data, \
                                 xml.getElementsByTagName('ymin')[0].firstChild.data, \
                                 xml.getElementsByTagName('ymax')[0].firstChild.data
    return [x_min, x_max, y_min, y_max]


class VidDataset(Dataset):
    """VID dataset."""
    def __init__(self, xml_annotations_dir, root_dir, transform=None):

        self.annotations = list_files(xml_annotations_dir)
        self.images = list_files(root_dir)
        self.box_frame = pd.DataFrame(columns=['img', 'x_min', 'x_max', 'y_min', 'y_max'])

        for img, ann in zip(self.images, self.annotations):
            df_row = [img] + parse_annotation(ann)
            self.box_frame = self.box_frame.append(pd.Series(df_row, index=['img', 'x_min', 'x_max', 'y_min', 'y_max']), ignore_index=True )
            print(df_row)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.box_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.box_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.box_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
