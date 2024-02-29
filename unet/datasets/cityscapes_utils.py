import logging
import random
from pathlib import Path
from typing import List, Tuple

def findCorrespondingMaskFromImage(img_filename_: str,
                                   masks_path_: str) -> str:
    masks_path = Path(masks_path_)
    
    if not masks_path.is_dir():
        logging.error("Given masks path is invalid!")
        return None
    
    # Retrieve the name of mask file
    mask_filename = "{}_{}".format(img_filename_.split("_leftImg8bit")[0],
                                   "gtFine_labelIds.png")
    
    logging.debug(f"mask_filename: {mask_filename}")
    
    ## REMARK: assuming path is given at "gtFine/" level!
    corr_mask_path = list(masks_path.glob("*/*/" + mask_filename))

    logging.debug(f"Found {len(corr_mask_path)} correspondings!")
    logging.debug(f"corr_mask_path: {corr_mask_path}")

    if len(corr_mask_path) > 0 :
        return str(corr_mask_path[0])
    else:
        logging.error("Haven't found corresponding mask in the given path!")
        return None

def findRandomImagesAndMasksPaths(images_path_: str,
                                  masks_path_: str,
                                  n_ : int,
                                  seed_: int = 42) -> List[Tuple[str, str]] :   
    corresponding_paths = []

    ## Convert strings to Path objects
    images_path = Path(images_path_)
    masks_path = Path(masks_path_)

    ## Check that the given paths are existing directories
    if not images_path.is_dir():
        logging.error("Given images path is invalid!")
        return None
    
    if not masks_path.is_dir():
        logging.error("Given masks path is invalid!")
        return None

    ## Find all images inside the given path
    ## REMARK: assuming paths are given at "leftImg8bit/" or
    ## "gtFine/" levels!
    images_path_list = list(images_path.glob("*/*/*.png"))
    masks_path_list = list(masks_path.glob("*/*/*_labelIds.png"))

    logging.info(f"Found {len(images_path_list)} images and "
                 f"{len(masks_path_list)} corresponding masks!")

    ## Perform some checks on found images
    if (len(images_path_list) == 0):
        logging.warning("No input images found!")
        return None
    
    if (len(masks_path_list) == 0):
        logging.warning("No masks found!")
        return None
    
    if (len(images_path_list) != len(masks_path_list)):
        logging.warning("Different number of images and masks!")
        return None
    
    ## Randomly find n image paths
    random.seed(seed_)
    random_image_paths = random.sample(images_path_list, k=n_)

    ## Find corresponding masks
    for random_img_path in random_image_paths:
        random_img_path = str(random_img_path)
        random_img_fname = random_img_path.split("/")[-1]

        logging.debug(f"random_img_fname: {random_img_fname}")

        ## Go looking for corresponding masks
        corr_mask_path = findCorrespondingMaskFromImage(img_filename_=random_img_fname,
                                                        masks_path_=masks_path_)
        
        if corr_mask_path is None:
            return None
        
        corresponding_paths.append(
            (random_img_path, corr_mask_path)
        )

    return corresponding_paths
