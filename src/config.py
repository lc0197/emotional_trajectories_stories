import os
from pathlib import Path
import torch

ROOT_DIR = Path(os.path.dirname(__file__)).parent.absolute()

DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

PREDICTION_DIR = os.path.join(ROOT_DIR, 'predictions')

RES_DIR = os.path.join(ROOT_DIR, 'res')
PRETRAIN_DIR = os.path.join(RES_DIR, 'pretraining')

device = 'cuda' if torch.cuda.is_available() else 'cpu'