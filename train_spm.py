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

import sentencepiece as spm

# train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.

spm.SentencePieceTrainer.train('--input=scan.txt --model_prefix=m --vocab_size=22  --model_type=word')# --unk_id=1 --bos_id=2 --eos_id=3 --user_defined_symbols=<s>,</s>')

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('m.model')

print('bos=', sp.bos_id())
print('eos=', sp.eos_id())
print('unk=', sp.unk_id())
print('pad=', sp.pad_id())  # disabled by default


# encode: text => id
print(sp.encode_as_ids('This is a test'))
print(sp.encode_as_ids('<s> walk run jump </s>'))
print(sp.decode_ids( sp.encode_as_ids('<s> walk run jump </s>') ))
print(sp.encode_as_pieces('<s> walk run jump </s>'))
print(sp.decode_pieces( sp.encode_as_pieces('<s> walk run jump </s>') ))
