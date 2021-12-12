import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_ROOT = "monet2photo/"
LEARNING_RATE = 1e-5
NUM_EPOCHS = 150
DEBUG = False
