import os
import logging
from typing import Optional, Callable, Tuple, Any
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomCityscapesDataset(Dataset):
    ## transform is a transformation function of the Albumentations
    ## augmentation pipeline
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] =None) -> None:
        super().__init__()

        self.root = root
        self.images_dir = os.path.join(self.root, "leftImg8bit", split)
        self.targets_dir = os.path.join(self.root, "gtFine", split)
        self.transform = transform

        self.images = []
        self.targets = []

        logging.debug(f"images_dir: {self.images_dir}")
        logging.debug(f"targets_dir: {self.targets_dir}")
        
        ## Loop over all images and corresponding targets
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                img_file_path = os.path.join(img_dir, file_name)
                
                logging.debug(img_file_path)

                target_name = "{}_{}".format(file_name.split("_leftImg8bit")[0], 
                                             "gtFine_labelIds.png")
                
                target_file_path = os.path.join(target_dir, target_name)
                logging.debug(target_file_path)

                self.images.append(img_file_path)
                self.targets.append(target_file_path)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        logging.debug(f"self.images[idx]: {self.images[idx]}")
        logging.debug(f"self.targets[idx]: {self.targets[idx]}")

        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.targets[idx])

        logging.debug(f"type(image): {type(image)}")
        logging.debug(f"type(mask): {type(mask)}")

        # Convert PIL images to Numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        logging.debug(f"type(image): {type(image)}")
        logging.debug(f"type(mask): {type(mask)}")

        logging.debug(f"image shape: {image.shape} - image type: {image.dtype}")
        logging.debug(f"mask shape: {mask.shape} - mask type: {mask.dtype}")

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


def test():
    print("Running test...")

    logging.basicConfig(level=logging.INFO)

    data_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    )

    custom_cityscapes_dataset = CustomCityscapesDataset(
        root="/home/andrea/datasets/cityscapes",
        split="train",
        transform=data_transform)
    
    logging.info(len(custom_cityscapes_dataset))

    img_sample, smnt_sample = custom_cityscapes_dataset[0]

    logging.info(f"type(img_sample): {type(img_sample)}")
    logging.info(f"type(smnt_sample): {type(smnt_sample)}")

if __name__ == "__main__":
    test()