import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_ROOT = "monet2photo/"
LEARNING_RATE = 2e-4
NUM_EPOCHS = 150
DEBUG = False
LAMBDA = 10
LAMBDA_IDENTITY = 0.5*LAMBDA
LOAD_CHECKPOINT = True