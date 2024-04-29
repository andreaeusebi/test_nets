## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

import sys 
sys.path.insert(0, "./../" )

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import hf_hub_download, login
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import TrainingArguments
from transformers import Trainer
import torch
from torch import nn
import evaluate

import json
import logging
import numpy as np
import matplotlib.pyplot as plt

import config
from segformer_huggingface.dataset_utils.get_dataset import getHfDataset
from segformer_huggingface.dataset_utils.label import getId2Label

metric = evaluate.load("mean_iou")

def train_transforms(example_batch):
    images = [x.convert("RGB") for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

def val_transforms(example_batch):
    images = [x.convert("RGB") for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels)
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
        metrics = metric._compute(
                predictions=pred_labels,
                references=labels,
                num_labels=len(id2label),
                ignore_index=255,
                reduce_labels=False, # we've already reduced the labels ourselves
            )

    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics

def get_seg_overlay(image, seg):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label_id, color in config.PALETTE.items():
        color_seg[seg == label_id, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    return img

def main():
    logging.basicConfig(format="[train.py][%(levelname)s]: %(message)s",
					    level=config.LOGGING_LEVEL)
    
    # Login to Huggingface
    login(config.HF_TOKEN)
    
    ## Loading the dataset ##

    train_ds = getHfDataset(dataset_names_=config.DATASETS, split_="train")
    valid_ds = getHfDataset(dataset_names_=config.DATASETS, split_="val")

    global id2label

    id2label = getId2Label(config.DATASETS[0], exclude_255_=True)
    
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)

    logging.info(f"Number of labels: {num_labels}")
    logging.info(f"Id2label: {id2label}")
    logging.info(f"label2id: {label2id}")

    ## Image processor & data augmentation ##
    global processor
    processor = SegformerImageProcessor(size= {"height": config.H, "width": config.W})

    # Set transforms
    train_ds.set_transform(train_transforms)
    valid_ds.set_transform(val_transforms)

    ## Fine-tune a SegFormer model ##
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.IN_MODEL_NAME,
        id2label=id2label,
        label2id=label2id
    )

    ## Set up the Trainer ##

    epochs = 2
    lr = 0.00006
    batch_size = 2

    hub_model_id = config.OUT_MODEL_NAME

    training_args = TrainingArguments(
        output_dir=config.OUT_MODEL_NAME,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=3,
        evaluation_strategy="epoch",
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

    trainer.train()
    
    kwargs = {
        "tags": ["vision", "image-segmentation"],
        "finetuned_from": config.IN_MODEL_NAME,
        "dataset(s)": config.HF_DATASET,
    }
    
    processor.push_to_hub(hub_model_id, private=True)
    trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    main()