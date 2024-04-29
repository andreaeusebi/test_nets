from collections import namedtuple
from typing import Dict, Tuple, List, Optional
import numpy as np

ACCEPTABLE_DATASETS_NAMES   = ["TMHMI", "Cityscapes"]

# Tuple representing a generic label
Label = namedtuple( 'Label' , [
    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class.

    'id'          , # An integer ID that is associated with this label.
                    # These are the IDs that represent labels in the ground truth images.
                    # An ID of -1 means that this label is not present in the dataset.
                    # Pay attention in modifying these labels.

    'trainId'     , # IDs used for training, i.e., these represent the classes that
                    # you want your model to predict. Original IDs are mapped to these
                    # IDs. Generally used to reduce the number of classes. In fact,
                    # multiple IDs can be mapped to the same trainId.
                    # Use '255' to indicate that the class must be ignored.

    'frequency'   , # Frequency as percentage of the label within the dataset.

    'color'       , # The color of this label as RGB values.
])

##### ----- TMHMI Labels ----- #####

LABELS_TMHMI = [
    #          name                   id    trainId     frequency          color
    Label(  'unlabeled'             ,  0 ,    255 ,      0.0    ,      (  0,  0,  0) ), #
    Label(  'road'                  ,  1 ,      0 ,      0.0    ,      (128, 64,128) ), #
    Label(  'sidewalk'              ,  2 ,      1 ,      0.0    ,      (244, 35,232) ), #
    Label(  'building'              ,  3 ,      2 ,      0.0    ,      ( 70, 70, 70) ), #
    Label(  'wall'                  ,  4 ,      3 ,      0.0    ,      (102,102,156) ), #
    Label(  'fence'                 ,  5 ,      4 ,      0.0    ,      (190,153,153) ), #
    Label(  'pole'                  ,  6 ,      5 ,      0.0    ,      (153,153,153) ), #
    Label(  'traffic_light'         ,  7 ,      6 ,      0.0    ,      (250,170, 30) ), #
    Label(  'traffic_sign'          ,  8 ,      7 ,      0.0    ,      (220,220,  0) ), #
    Label(  'vegetation'            ,  9 ,      8 ,      0.0    ,      (107,142, 35) ), #
    Label(  'terrain'               , 10 ,      9 ,      0.0    ,      (152,251,152) ), #
    Label(  'sky'                   , 11 ,     10 ,      0.0    ,      ( 70,130,180) ), #
    Label(  'person'                , 12 ,     11 ,      0.0    ,      (220, 20, 60) ), #
    Label(  'rider'                 , 13 ,     12 ,      0.0    ,      (255,  0,  0) ), #
    Label(  'car'                   , 14 ,     13 ,      0.0    ,      (  0,  0,142) ), #
    Label(  'truck'                 , 15 ,     14 ,      0.0    ,      (  0,  0, 70) ), #
    Label(  'bus'                   , 16 ,     15 ,      0.0    ,      (  0, 60,100) ), #
    Label(  'forklift'              , 17 ,     20 ,      0.0    ,      (220,220,220) ), #
    Label(  'motorcycle'            , 18 ,     17 ,      0.0    ,      (  0,  0,230) ), #
    Label(  'crane'                 , 19 ,    255 ,      0.0    ,      ( 80,227,194) ), #
    Label(  'rack'                  , 20 ,     19 ,      0.0    ,      (248,248, 28) ), #
    Label(  'container'             , 21 ,    255 ,      0.0    ,      ( 81,  0, 81) ), #
    Label(  'pallet'                , 22 ,     21 ,      0.0    ,      (189, 16,224) ), #
    Label(  'object'                , 23 ,     22 ,      0.0    ,      (114, 74,  0) ), #
    Label(  'mast'                  , 24 ,    255 ,      0.0    ,      ( 65,117,  5) ), #
    Label(  'train'                 , -1 ,     16 ,      0.0    ,      (  0, 80,100) ), #
    Label(  'bicycle'               , -1 ,     18 ,      0.0    ,      (119, 11, 32) )  #
]   

###################################################

##### ----- Cityscapes Labels ----- #####

LABELS_CS = [
    #          name                   id    trainId     frequency          color
    Label(  'unlabeled'             ,  0 ,    255 ,      0.0001 ,      (  0,  0,  0) ), #
    Label(  'ego vehicle'           ,  1 ,    255 ,      0.0458 ,      (  0,  0,  0) ), #
    Label(  'rectification border'  ,  2 ,    255 ,      0.0130 ,      (  0,  0,  0) ), #
    Label(  'out of roi'            ,  3 ,    255 ,      0.0150 ,      (  0,  0,  0) ), #
    Label(  'static'                ,  4 ,    255 ,      0.0134 ,      (  0,  0,  0) ), #
    Label(  'dynamic'               ,  5 ,    255 ,      0.0028 ,      (111, 74,  0) ), #
    Label(  'ground'                ,  6 ,    255 ,      0.0121 ,      ( 81,  0, 81) ), #
    Label(  'road'                  ,  7 ,      0 ,      0.3264 ,      (128, 64,128) ), #
    Label(  'sidewalk'              ,  8 ,      1 ,      0.0538 ,      (244, 35,232) ), #
    Label(  'parking'               ,  9 ,    255 ,      0.0062 ,      (250,170,160) ), #
    Label(  'rail track'            , 10 ,    255 ,      0.0018 ,      (230,150,140) ), #
    Label(  'building'              , 11 ,      2 ,      0.2020 ,      ( 70, 70, 70) ), #
    Label(  'wall'                  , 12 ,      3 ,      0.0058 ,      (102,102,156) ), #
    Label(  'fence'                 , 13 ,      4 ,      0.0077 ,      (190,153,153) ), #
    Label(  'guard rail'            , 14 ,    255 ,      0.0001 ,      (180,165,180) ), #
    Label(  'bridge'                , 15 ,    255 ,      0.0028 ,      (150,100,100) ), #
    Label(  'tunnel'                , 16 ,    255 ,      0.0005 ,      (150,120, 90) ), #
    Label(  'pole'                  , 17 ,      5 ,      0.0108 ,      (153,153,153) ), #
    Label(  'polegroup'             , 18 ,    255 ,      0.0001 ,      (153,153,153) ), #
    Label(  'traffic light'         , 19 ,      6 ,      0.0018 ,      (250,170, 30) ), #
    Label(  'traffic sign'          , 20 ,      7 ,      0.0048 ,      (220,220,  0) ), #
    Label(  'vegetation'            , 21 ,      8 ,      0.1410 ,      (107,142, 35) ), #
    Label(  'terrain'               , 22 ,      9 ,      0.0102 ,      (152,251,152) ), #
    Label(  'sky'                   , 23 ,     10 ,      0.0355 ,      ( 70,130,180) ), #
    Label(  'person'                , 24 ,     11 ,      0.0107 ,      (220, 20, 60) ), #
    Label(  'rider'                 , 25 ,     12 ,      0.0012 ,      (255,  0,  0) ), #
    Label(  'car'                   , 26 ,     13 ,      0.0619 ,      (  0,  0,142) ), #
    Label(  'truck'                 , 27 ,     14 ,      0.0023 ,      (  0,  0, 70) ), #
    Label(  'bus'                   , 28 ,     15 ,      0.0020 ,      (  0, 60,100) ), #
    Label(  'caravan'               , 29 ,    255 ,      0.0004 ,      (  0,  0, 90) ), #
    Label(  'trailer'               , 30 ,    255 ,      0.0002 ,      (  0,  0,110) ), #
    Label(  'train'                 , 31 ,     16 ,      0.0020 ,      (  0, 80,100) ), #
    Label(  'motorcycle'            , 32 ,     17 ,      0.0008 ,      (  0,  0,230) ), #
    Label(  'bicycle'               , 33 ,     18 ,      0.0036 ,      (119, 11, 32) ), #
    Label(  'license plate'         , -1 ,    255 ,      0.0    ,      (  0,  0,142) )  #
]   

###################################################

##### ----- Methods using labels ----- #####

def labelIDsToTrainIds(orig_mask_, label_metatada_):
    """ 
    Remap the IDs of the given segmentation bitmap from the original
    label IDs to the corresponding train IDs.
    The semantic bitmaps are torch tensors.
    """
    # Initialize new label mask as a copy of the original one 
    new_mask = orig_mask_.clone().detach()

    ## Remap the original labels to the new set of labels
    for label in label_metatada_:      
        original_id = label.id
        new_id = label.trainId
       
        new_mask[orig_mask_ == original_id] = new_id

    return new_mask

def labelsIDsToTrainIdsOnArrays(semantic_bitmap_: np.array,
                                label_metatada_: List[Label]):
    """ 
    Remap the IDs of the given segmentation bitmap from the original
    label IDs to the corresponding train IDs.
    The semantic bitmaps are Numpy arrays.
    """
    
    # Initialize new semantic bitmap as a copy of the original one
    mapped_semantic_bitmap = np.copy(semantic_bitmap_)

    ## Remap the original labels to the new set of labels
    for label in label_metatada_:
        original_id = label.id
        new_id = label.trainId
        mapped_semantic_bitmap[semantic_bitmap_ == original_id] = new_id

    return mapped_semantic_bitmap

def getPalette(dataset_: str) -> Dict[int, Tuple[int, int, int]]:
    assert dataset_ in ACCEPTABLE_DATASETS_NAMES

    if dataset_ == "Cityscapes":
        labels = LABELS_CS
    elif dataset_ == "TMHMI":
        labels = LABELS_TMHMI

    palette = {}

    for label in labels:
        # Check if another label with same trainId has already been added
        # (in case of multiple labels with same trainId, the color of the
        # first one is used)
        if label.trainId in palette:
            continue
        # If trainId value is 255, ignore that label
        elif label.trainId == 255:
            continue
        # We can add the label to the palette
        else:
            palette_pair = {label.trainId : label.color}
            palette.update(palette_pair)

    return palette

def getId2Label(dataset_: str, exclude_255_: Optional[bool] = True) -> Dict[int, str]:
    assert dataset_ in ACCEPTABLE_DATASETS_NAMES
    
    id2label = {}

    if dataset_ == "Cityscapes":
        labels = LABELS_CS
    elif dataset_ == "TMHMI":
        labels = LABELS_TMHMI

    for label in labels:
        if label.trainId in id2label:
            continue
        elif exclude_255_ and label.trainId == 255:
            continue
        else:
            new_pair = {label.trainId : label.name}
            id2label.update(new_pair)  

    return id2label
###################################################
