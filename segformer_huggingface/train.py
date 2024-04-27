## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

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
from dataset_utils.get_dataset import getHfDataset

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
    palette = np.array(config.PALETTE_FUNCTION())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

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

    train_ds = getHfDataset(dataset_names_=["TMHMI", "Cityscapes"], split_="train")
    valid_ds = getHfDataset(dataset_names_=["TMHMI", "Cityscapes"], split_="val")

    global id2label

    filename = "dataset_utils/id2label.json"
    # id2label = json.load(
    #     open(hf_hub_download(repo_id=config.HF_DATASET,
    #                          filename=filename,
    #                          repo_type="dataset"),
    #     "r")
    # )

    id2label = json.load(open(filename))
    
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)

    logging.info(f"Number of labels: {num_labels}")
    logging.info(f"Id2label: {id2label}")
    logging.info(f"label2id: {label2id}")

    ## Image processor & data augmentation ##
    global processor
    # processor = SegformerImageProcessor(do_reduce_labels=True)
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
        "dataset": config.HF_DATASET,
    }
    
    processor.push_to_hub(hub_model_id, private=True)
    trainer.push_to_hub(**kwargs)

    exit()
    
    ## Test inference
    from PIL import Image
    image = Image.open(("/home/andrea/my_projects/test_nets/segformer_huggingface/segments/"
                        "andrea_eusebi_TMHMI_Semantic_Dataset/v0.1/seq_04_frame000000_rgb.png")).convert("RGB")
   
    inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    print(f"logits.shape: {logits.shape}")

    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1], # (height, width)
        mode='bilinear',
        align_corners=False
    )

    # Second, apply argmax on the class dimension
    pred_seg = upsampled_logits.argmax(dim=1)[0].to("cpu")

    pred_img = get_seg_overlay(image, pred_seg)
    # gt_img = get_seg_overlay(image, np.array(gt_seg))

    # f, axs = plt.subplots(1, 2)
    # f.set_figheight(30)
    # f.set_figwidth(50)

    # axs[0].set_title("Prediction", {'fontsize': 40})
    # axs[0].imshow(pred_img)
    # axs[1].set_title("Ground truth", {'fontsize': 40})
    # axs[1].imshow(gt_img)

    plt.figure(figsize=(15, 10))
    plt.imshow(pred_img)

    plt.show()

if __name__ == "__main__":
    main()