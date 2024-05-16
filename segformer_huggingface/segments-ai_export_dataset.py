## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
from segments.utils import get_semantic_bitmap
from segments.huggingface import release2dataset

import matplotlib.pyplot as plt

API_KEY         = "e6eb70a8f4cd51d900b5ca6a0fcbb504070b5307"
DATASET_ID      = "andrea_eusebi/TMHMI_Semantic_Dataset"
RELEASE_NAME    = "v0.1"
EXPORT_FORMAT   = "semantic"
VISUALIZE_DS    = False

# Initialize a SegmentsDataset from the release file
segments_client = SegmentsClient(API_KEY)
segments_release = segments_client.get_release(DATASET_ID, RELEASE_NAME)
segments_dataset = SegmentsDataset(segments_release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

# Export to given format
export_dataset(segments_dataset, export_format=EXPORT_FORMAT)

if VISUALIZE_DS:
    for sample in segments_dataset:
        # Print the sample name and list of labeled objects
        print(sample['name'])
        print(sample['annotations'])
        
        fig = plt.figure()
        
        # Show the image
        fig.add_subplot(1, 2, 1)
        plt.imshow(sample['image'])
        plt.title("Original Image")
        
        # # Show the instance segmentation label
        # fig.add_subplot(1, 3, 2)
        # plt.imshow(sample['segmentation_bitmap'])
        # plt.title("Instance Segmentation bitmap")
        
        # Show the semantic segmentation label
        fig.add_subplot(1, 2, 2)
        semantic_bitmap = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])
        plt.imshow(semantic_bitmap)
        plt.title("Semantic Segmentation bitmap")

        plt.show()

# Convert it to a Huggingface Dataset
hf_dataset = release2dataset(segments_release)

# The returned object is an Huggingface Dataset
print(f"hf_dataset:\n{hf_dataset}")
print(f"features:\n{hf_dataset.features}")