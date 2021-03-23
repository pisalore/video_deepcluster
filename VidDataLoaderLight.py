from pathlib import Path

from skimage import io
import torch
from torchvision.datasets import ImageFolder


class VidDatasetLight(ImageFolder):
    """VID dataset Light.
        This dataset is made starting from the original Vid Dataset, assuming that images have been already cropped
        (for example, using an offline preprocessing procedure) trying to speed up vid classification.
    """

    def __init__(self, root_dir: str, transform=None, labels=None, crop_coords=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            labels data labels from VID annotations
            crop_coords a dictionary with image crop coordinates from VID annotations
        """
        super().__init__(root_dir, transform)
        self.vid_labels = labels
        self.crop_coords = crop_coords

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx][0]
        image = torch.from_numpy(io.imread(img_name).astype('float32') / 255)
        if self.vid_labels is not None:
            label = self.vid_labels[img_name]
        else:
            label = -1
        sample = {'image': image,
                  'name': self.imgs[idx][0],
                  'label': int(label) - 1}  # classes are from 0 to 29

        if self.transform:
            sample = self.transform(sample)
        return sample
