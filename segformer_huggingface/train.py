## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

import sys 
sys.path.insert(0, "./../" )

from huggingface_hub import login, upload_file
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import TrainingArguments
from transformers import Trainer
import torch
from torch import nn
import evaluate
from torchmetrics.classification import MulticlassConfusionMatrix
import albumentations as Albu
import cv2

import logging
import numpy as np
import matplotlib.pyplot as plt

## Local Imports ##
import config
from segformer_huggingface.dataset_utils.get_dataset import getHfDataset
from segformer_huggingface.dataset_utils.label import getId2Label

### ---- Global Variables ----- ###
g_epoch_counter = 0

### ---- Training Metrics ----- ###
mean_iou = evaluate.load("mean_iou")

valid_conf_matrix_metric = MulticlassConfusionMatrix(
    num_classes=23,
    normalize="true",
    ignore_index=255
)

### ---- Augmentations ----- ###
train_augm = Albu.Compose(
    [
        Albu.OneOf(
            [
                # 960 because it half of TMHMI images width
                Albu.RandomSizedCrop(min_max_height=(config.H/2, 960),
                                     size=(config.H, config.W),
                                     w2h_ratio=config.W/config.H,
                                     p=0.5,
                                     interpolation=cv2.INTER_AREA),
                Albu.Resize(config.H,
                            config.W,
                            p=0.5,
                            interpolation=cv2.INTER_AREA),
            ],
            p=1.0
        ),
        Albu.HorizontalFlip(p=0.5),
        Albu.ColorJitter(brightness=0.25,
                         contrast=0.25,
                         saturation=0.25,
                         hue=0.1,
                         p=0.5),
        Albu.ChannelShuffle(p=0.2)
    ]
)

valid_augm = Albu.Compose(
    [
        Albu.Resize(config.H,
                    config.W,
                    interpolation=cv2.INTER_AREA,
                    p=1.0)
    ]
)

def train_transforms(batch):
    assert(len(batch['pixel_values']) == len(batch['label']))

    ### ----- Approach using SegformerImageProcessor ----- ###
    # images = [x.convert("RGB") for x in batch['pixel_values']]
    # labels = [x for x in batch['label']]
    # inputs = g_processor(images, labels)

    ### ----- Approach using Albumentations Transformations ----- ###
    images = []
    labels = []

    # Parse and augments both images and labels
    for (img_pil, label_pil) in zip(batch['pixel_values'], batch['label']):
        transformed = train_augm(image=np.array(img_pil.convert("RGB")),
                                 mask=np.array(label_pil))
        images.append(transformed["image"])
        labels.append(transformed["mask"])

    assert(len(images) == len(labels))

    # Complete preprocessing (normalization and rescaling to 0-1 range)
    inputs = g_processor(images, labels)

    # inputs = g_processor.preprocess(images=images,
    #                               segmentation_maps=labels,
    #                               return_tensors="pt")

    ## Debug
    logging.debug(f"type(images):                    {type(images)}")
    logging.debug(f"type(labels):                    {type(labels)}")
    logging.debug(f"type(images[0]):                 {type(images[0])}")
    logging.debug(f"type(labels[0]):                 {type(labels[0])}")
    logging.debug(f"type(inputs):                    {type(inputs)}")
    logging.debug(f"type(inputs['pixel_values']):    {type(inputs['pixel_values'])}")
    logging.debug(f"type(inputs['labels']):          {type(inputs['labels'])}")
    logging.debug(f"type(inputs['pixel_values'][0]): {type(inputs['pixel_values'][0])}")
    logging.debug(f"type(inputs['labels'][0]):       {type(inputs['labels'][0])}")

    return inputs

def val_transforms(batch):
    assert(len(batch['pixel_values']) == len(batch['label']))

    ### ----- Approach using SegformerImageProcessor ----- ###
    # images = [x.convert("RGB") for x in batch['pixel_values']]
    # labels = [x for x in batch['label']]
    # inputs = g_processor(images, labels)

    ### ----- Approach using Albumentations Transformations ----- ###
    images = []
    labels = []

    # Parse and augments both images and labels
    for (img_pil, label_pil) in zip(batch['pixel_values'], batch['label']):
        transformed = valid_augm(image=np.array(img_pil.convert("RGB")),
                                 mask=np.array(label_pil))
        images.append(transformed["image"])
        labels.append(transformed["mask"])

    assert(len(images) == len(labels))

    # Complete preprocessing (normalization and rescaling to 0-1 range)
    inputs = g_processor(images, labels)

    return inputs

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()

        # Inform Python that g_epoch_counter is a global var (needed to change its value)
        global g_epoch_counter

        ## Compute confusion matrix if this is last epoch
        if g_epoch_counter == config.EPOCHS - 1:
            logging.info(f"Arrived at last epoch ({g_epoch_counter}). Computing confusion matrix!")
            
            labels_tensor = torch.from_numpy(labels)
            valid_conf_matrix_metric(logits_tensor, labels_tensor)

            fig, ax = valid_conf_matrix_metric.plot(labels=g_id2label.values())
            figure = plt.gcf() # get current figure
            figure.set_size_inches(16, 16) # set figure's size manually

            plt.savefig(config.PROJECT_DIR + config.OUT_MODEL_NAME + "/" + "conf_matrix.png",
                        dpi=200,
                        bbox_inches='tight') # bbox_inches removes extra white spaces

            figure.show()
            plt.show()

        mean_iou_results = mean_iou._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(g_id2label),
            ignore_index=255,
            reduce_labels=False, # we've already reduced the labels ourselves
        )

    # add per category metrics as individual key-value pairs
    per_category_accuracy = mean_iou_results.pop("per_category_accuracy").tolist()
    per_category_iou = mean_iou_results.pop("per_category_iou").tolist()

    mean_iou_results.update({f"accuracy_{g_id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    mean_iou_results.update({f"iou_{g_id2label[i]}": v for i, v in enumerate(per_category_iou)})

    ## REMARK: THIS WORKS AS LONG AS WE RUN THIS METHOD ONCE PER EPOCH!!
    g_epoch_counter += 1

    return mean_iou_results

def get_seg_overlay(image, seg):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label_id, color in config.PALETTE.items():
        color_seg[seg == label_id, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    return img

def appendContentInFile(file_name:str, line_idx:int, content:str):
    """
    This method add content at a specific line to the given file.  
    """
    # Python starts counting at 0, but people start counting at one. This accounts for that.
    line_idx -= 1

    with open(file_name, "r") as file: # Open the file in read mode
        # Assign the file as a list to a variable
        lines = file.readlines()
        # Save the line in a temporayry string
        prev_line = lines[line_idx]
        # Append the proper line with the provided content
        lines[line_idx] = content + "\n" + prev_line

    with open(file_name, "w") as file: # Open the file in write mode
        file.write("".join(lines)) # Write the modified content to the file

def main():
    logging.basicConfig(format="[train.py][%(levelname)s]: %(message)s",
					    level=config.LOGGING_LEVEL)
    
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logging.info("Starting training script...")
    
    # Login to Huggingface
    login(config.HF_TOKEN)
    
    ## Loading the dataset(s) ##

    train_ds = getHfDataset(dataset_names_=config.DATASETS, split_="train")
    valid_ds = getHfDataset(dataset_names_=config.DATASETS, split_="val")

    global g_id2label

    g_id2label = getId2Label(config.DATASETS[0], exclude_255_=True)
    
    g_id2label = {int(k): v for k, v in g_id2label.items()}
    label2id = {v: k for k, v in g_id2label.items()}

    num_labels = len(g_id2label)

    logging.info(f"Number of labels: {num_labels}")
    logging.info(f"Id2label: {g_id2label}")
    logging.info(f"label2id: {label2id}")

    ## Image processor & data augmentation ##
    global g_processor
    # g_processor = SegformerImageProcessor(size= {"height": config.H, "width": config.W})
    g_processor = SegformerImageProcessor(do_resize=False,
                                          do_rescale=True,
                                          do_normalize=True,
                                          do_reduce_labels=False)

    # Set transforms
    train_ds.set_transform(train_transforms)
    valid_ds.set_transform(val_transforms)

    ## Fine-tune a SegFormer model ##
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.IN_MODEL_NAME,
        id2label=g_id2label,
        label2id=label2id
    )

    ## Set up the Trainer ##

    epochs          = config.EPOCHS
    lr              = config.LEARNING_RATE
    batch_size      = config.BATCH_SIZE
    hub_model_id    = config.OUT_MODEL_NAME

    training_args = TrainingArguments(
        output_dir=config.OUT_MODEL_NAME,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=3,
        evaluation_strategy="epoch",  ## NEED TO DO EVERY EPOCH OTHERWISE CONFUS MAT COUNTER DOESN'T WORK
        save_strategy="epoch",
        # save_steps=20,
        # eval_steps=20,
        logging_steps=1,
        eval_accumulation_steps=5,
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_strategy="every_save",
        hub_private_repo=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    ## Training
    trainer.train()
    
    ## Saving Results to HuggingFace Hub
    kwargs = {
        "tags": ["vision", "image-segmentation"],
        "finetuned_from": config.IN_MODEL_NAME,
        "dataset": config.HF_DATASET,
    }
    
    logging.info("[train.py]: Pushing results to Huggingface hub!")

    g_processor.push_to_hub(hub_model_id, private=True)
    trainer.push_to_hub(**kwargs)

    ## Add Confusion Matrix to model card
    logging.info("[train.py]: Adding confusions matrix to model hub.")

    appendContentInFile(
        config.PROJECT_DIR + config.OUT_MODEL_NAME + "/" + "README.md",
        line_idx=71,  # THIS WORKS AS LONG AS THE README DOESN'T CHANGE!!
        content="## Confusion Matrix\n![Confusion Matrix on validation data ](conf_matrix.png)\n"
    )
    
    with open(config.PROJECT_DIR + config.OUT_MODEL_NAME + "/" + "README.md", "rb") as fobj:
        upload_file(path_or_fileobj=fobj,
                    path_in_repo="README.md",
                    repo_id=config.HF_USERNAME + "/" + config.OUT_MODEL_NAME,
                    token=config.HF_TOKEN,
                    repo_type="model",
                    commit_message="TEST")

    logging.info("[train.py]: Training script completed!")

if __name__ == "__main__":
    main()