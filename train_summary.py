# based on https://blog.paperspace.com/generating-text-summaries-gpt-2/

import argparse
from datetime import datetime
import json
import os
import pickle
import random
import sys
import time

import numpy as np
from pytorch_transformers import GPT2Tokenizer
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tnrange, tqdm
from pytorch_transformers import ConstantLRSchedule, GPT2Config, GPT2LMHeadModel,AdamW, GPT2Tokenizer, WarmupLinearSchedule
from tensorboardX import SummaryWriter


class GPT21024Dataset(Dataset):


    def __init__(self, root_dir, ids_file, mode='train',length=None):
        self.root_dir = root_dir
        self.tokenizer = add_special_tokens()
        with open(ids_file,'r') as f:
            if mode=='train':
                self.idxs = json.load(f)['train_ids']
            elif mode=='valid':
                self.idxs = json.load(f)['valid_ids']
            else:
                self.idxs = json.load(f)['test_ids']
        if len == None:
            self.len = len(self.idxs)
        else:
            self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        idx = self.idxs[idx]
        file_name = os.path.join(self.root_dir,str(idx)+".json")
        with open(file_name,'r') as f:
              data = json.load(f)
        text = self.tokenizer.encode(self.tokenizer.pad_token)*1024
        content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['abstract']
        text[:len(content)] = content
        text = torch.tensor(text)
        sample = {'article': text, 'sum_idx': len(data['article'])}
        return sample