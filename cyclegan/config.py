import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_ROOT = "../gan-getting-started"
LEARNING_RATE = 1e-5
NUM_EPOCHS = 150
