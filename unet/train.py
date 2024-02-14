
from euse_unet import EuseUnet
import utils
import datasets.cityscapes_dataset as cityscapes_dataset

import logging
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Resize

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

## Params
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 34
BATCH_SIZE = 32

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        print(f"--- Beginning of batch {batch}. ---")

        utils.logTensorInfo(X, "X")
        utils.logTensorInfo(y, "y")

        X = Resize(size=(572, 572), antialias=True)(X)
        y = Resize(size=(572, 572), antialias=True)(y)

        utils.logTensorInfo(X, "X")
        utils.logTensorInfo(y, "y")

        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

# def test_step(data_loader: torch.utils.data.DataLoader,
#               model: torch.nn.Module,
#               loss_fn: torch.nn.Module,
#               accuracy_fn,
#               device: torch.device = device):
#     test_loss, test_acc = 0, 0
#     model.to(device)
#     model.eval() # put model in eval mode
#     # Turn on inference context manager
#     with torch.inference_mode(): 
#         for X, y in data_loader:
#             # Send data to GPU
#             X, y = X.to(device), y.to(device)
            
#             # 1. Forward pass
#             test_pred = model(X)
            
#             # 2. Calculate loss and accuracy
#             test_loss += loss_fn(test_pred, y)
#             test_acc += accuracy_fn(y_true=y,
#                 y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
#             )
        
#         # Adjust metrics and print out
#         test_loss /= len(data_loader)
#         test_acc /= len(data_loader)
#         print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def main():
    logging.basicConfig(format="[train.py][%(levelname)s]: %(message)s",
                        level=logging.INFO)

    logging.debug("--- main() started. ---")

    ## 1. Load training and validation dataset
    train_data = Cityscapes(root="/home/andrea/datasets/cityscapes",
                            split="train",
                            mode="fine",
                            target_type="semantic",
                            transform=ToTensor(),
                            target_transform=utils.PILToTensor())
    
    val_data = Cityscapes(root="/home/andrea/datasets/cityscapes",
                          split="val",
                          mode="fine",
                          target_type="semantic",
                          transform=ToTensor(),
                          target_transform=utils.PILToTensor())
    
    logging.info(f"train images: {len(train_data.images)}")
    logging.info(f"train targets: {len(train_data.targets)}")
    logging.info(f"validation images: {len(val_data.images)}")
    logging.info(f"validation targets: {len(val_data.targets)}")
    
    img_sample, smnt_sample = train_data[0]
    
    utils.logTensorInfo(img_sample, "img_sample")
    utils.logTensorInfo(smnt_sample, "smnt_sample")

    ## 2. Prepare DataLoader   
    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(train_data,               # dataset to turn into iterable
                                  batch_size=BATCH_SIZE,    # how many samples per batch? 
                                  shuffle=True)             # shuffle data every epoch?
    
    val_dataloader = DataLoader(val_data,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    
    # Let's check out what we've created
    logging.info(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    logging.info(f"Length of val dataloader: {len(val_dataloader)} batches of {BATCH_SIZE}")

    # Check out what's inside the training dataloader
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    logging.info(f"train_features_batch.shape: {train_features_batch.shape}")
    logging.info(f"train_labels_batch.shape: {train_labels_batch.shape}")
    utils.logTensorInfo(train_features_batch, "train_features_batch")
    utils.logTensorInfo(train_labels_batch, "train_labels_batch")

    ## 3. Instantiate the model
    euse_unet = EuseUnet(input_channels_=INPUT_CHANNELS,
                         output_channels_=OUTPUT_CHANNELS).to(device)
    
    ## 4. Train the model

    # Setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=euse_unet.parameters(), 
                                lr=0.1)
    
    epochs = 1
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_step(model=euse_unet,
                   data_loader=train_dataloader,  
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn
        )
        # test_step(data_loader=test_dataloader,
        #     model=model_1,
        #     loss_fn=loss_fn,
        #     accuracy_fn=accuracy_fn
        # )

    logging.debug("--- main() completed. ---")

if __name__ == "__main__":
    main()