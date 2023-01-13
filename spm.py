import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import AdamW, GPT2Tokenizer, GPT2LMHeadModel
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import re
import pandas as pd
import torch.nn as nn
import numpy as np
import learn2learn as l2l
import Lang

scan_ds = load_dataset('scan', 'simple')

seq_vocab = Lang.Lang('seq_vocab')
cmd_vocab = Lang.Lang('cmd_vocab')

flcmd = open('scan_cmds_text.txt', 'w')
flact = open('scan_acts_text.txt', 'w')


for sentence in scan_ds['train']['commands']:
    flcmd.write(sentence+'\n')
    seq_vocab.add_sentence(sentence)

for sentence in scan_ds['test']['commands']:
    flcmd.write(sentence+'\n')
    seq_vocab.add_sentence(sentence)

for cmd in scan_ds['train']['actions']:
    flact.write(cmd+'\n')
    cmd_vocab.add_sentence(cmd)

for cmd in scan_ds['test']['actions']:
    flact.write(cmd+'\n')
    cmd_vocab.add_sentence(cmd)

flcmd.close()
flact.close()

seq_vocab.print()
cmd_vocab.print()