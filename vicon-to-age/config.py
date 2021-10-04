import torch

RUN_NAME = "run_02"
BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
EMBEDDING_DIMS = 1024
CHECKPOINT_PATH = 'checkpoints/run_02.pth.tar'
LOAD_CHECKPOINT = False
OPTIMIZER = 'Adam'
NUM_EPOCHS = 200
LEARNING_RATE = 0.001