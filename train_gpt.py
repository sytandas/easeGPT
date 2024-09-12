from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional

from tinygrad import Torch

@dataclass
class GPTconfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.modules): 
    def __init__(self, config):
        super().__init__()
        self.config = config 