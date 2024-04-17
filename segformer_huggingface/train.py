## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

from datasets import load_dataset
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
                ignore_index=0,
                reduce_labels=processor.do_reduce_labels,
            )

    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics

def tmhmi_palette():
    """Cityscapes palette that maps each class to RGB values."""
    return [[128, 64,128], #  0 
            [244, 35,232], #  1
            [ 70, 70, 70], #  2
            [102,102,156], #  3
            [190,153,153], #  4
            [153,153,153], #  5
            [250,170, 30], #  6
            [220,220,  0], #  7
            [107,142, 35], #  8
            [152,251,152], #  9
            [ 70,130,180], # 10
            [220, 20, 60], # 11
            [255,  0,  0], # 12
            [  0,  0,142], # 13
            [  0,  0, 70], # 14
            [  0, 60,100], # 15
            [220,220,220], # 16
            [  0,  0,230], # 17
            [ 80,227,194], # 18
            [248,248, 28], # 19
            [ 81,  0, 81], # 20
            [250,170,160], # 21
            [111, 74,  0], # 22
            [ 35, 35, 35], # 23
            [  0,  0,  0] # 255
           ]

def get_seg_overlay(image, seg):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(tmhmi_palette())
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
    login("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")
    
    ## Loading and preparing the dataset ##

    ds = load_dataset(f"{config.HF_USERNAME}/{config.DATASET_ID}")

    ds = ds.shuffle(seed=1)
    ds = ds["train"].train_test_split(test_size=0.2)
    train_ds = ds["train"]
    test_ds = ds["test"]

    logging.info(f"test_ds: {test_ds}")
    logging.info(f"test_ds[0]: {test_ds[0]}")
    logging.info(f"img: {test_ds[0]['pixel_values']}")
    logging.info(f"img type: {type(test_ds[0]['pixel_values'])}")
    logging.info(f"label: {test_ds[0]['label']}")
    logging.info(f"label type: {type(test_ds[0]['label'])}")

    logging.info(f"Lenght of training dataset: {len(train_ds)}")
    logging.info(f"Lenght of testing dataset: {len(test_ds)}")

    global id2label

    filename = "id2label.json"
    id2label = json.load(
        open(hf_hub_download(repo_id=f"{config.HF_USERNAME}/{config.DATASET_ID}",
                             filename=filename,
                             repo_type="dataset"),
        "r")
    )
    
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)

    logging.info(f"Number of labels: {num_labels}")
    logging.info(f"Id2label: {id2label}")
    logging.info(f"label2id: {label2id}")

    ## Image processor & data augmentation ##
    global processor
    processor = SegformerImageProcessor()

    # Set transforms
    train_ds.set_transform(train_transforms)
    test_ds.set_transform(val_transforms)

    ## Fine-tune a SegFormer model ##
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.IN_MODEL_NAME,
        id2label=id2label,
        label2id=label2id
    )

    ## Set up the Trainer ##

    epochs = 100
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
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        logging_steps=1,
        eval_accumulation_steps=5,
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_strategy="end",
        hub_private_repo=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    kwargs = {
        "tags": ["vision", "image-segmentation"],
        "finetuned_from": config.IN_MODEL_NAME,
        "dataset": f"{config.HF_USERNAME}/{config.DATASET_ID}",
    }
    
    processor.push_to_hub(hub_model_id, private=True)
    trainer.push_to_hub(**kwargs)

    exit()

    ## Test inference
    # ds = load_dataset(f"{config.HF_USERNAME}/{config.DATASET_ID}")

    # ds = ds.shuffle(seed=1)
    # ds = ds["train"].train_test_split(test_size=0.2)
    # train_ds = ds["train"]
    # test_ds = ds["test"]

    # print(f"test_ds: {test_ds}")
    # print(f"test_ds[0]: {test_ds[0]}")
    # print(f"img: {test_ds[0]['pixel_values']}")
    # print(f"img type: {type(test_ds[0]['pixel_values'])}")
    # print(f"label: {test_ds[0]['label']}")
    # print(f"label type: {type(test_ds[0]['label'])}")

    # image = test_ds[0]['pixel_values']
    # gt_seg = test_ds[0]['label']

    from PIL import Image
    image = Image.open(("/home/andrea/my_projects/test_nets/segformer_huggingface/segments/"
                        "andrea_eusebi_TMHMI_Semantic_Dataset/v0.1/seq_04_frame000000_rgb.png")).convert("RGB")

    processor = SegformerImageProcessor(size= {"height": config.H, "width": config.W})
    
    inputs = processor(images=image, return_tensors="pt").to(config.DEVICE)
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

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