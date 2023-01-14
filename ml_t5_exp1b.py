import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AdamW
import os
import re
import pandas as pd
import torch.nn as nn
import numpy as np
import learn2learn as l2l
import Lang

# read file names in directory
directory = '../SCAN/simple_split/size_variations/'
paths = []
for file in os.listdir(directory):
    if file.endswith(".txt"):
        paths.append( os.path.join(directory, file) )

paths.sort()

def parse_filename(fname):
    return int( re.search(r'.*p(\d+)\.txt', fname).group(1) )

combined_paths = []
processed = []
for i, p in enumerate(paths):
    num_i = parse_filename(p)

    if num_i in processed:
        continue

    for j, pj in enumerate(paths):
        if i == j:
            continue

        num_j = parse_filename(pj)
        if num_i == num_j:
            processed.append(num_i)
            combined_paths.append( (pj, p) )
            break

datasets = {}

def split_data_set(data):
    X = []
    y = []
    for d in data:
        match = re.search( r'IN: (.*) OUT: (.*)', d )
        X.append( match.group(1) )
        y.append( match.group(2) )

    return pd.DataFrame( {'Seq': X, 'Cmds': y } )

for trainset, testset in combined_paths:
    train_lines = open( trainset, 'r' ).readlines()
    test_lines = open( testset, 'r' ).readlines()

    file_id = parse_filename(trainset)
    datasets[file_id] = (split_data_set(train_lines), split_data_set(test_lines))

for k in datasets.keys():
    print(f'{k}: {len(datasets[k][0])} {len(datasets[k][1])}')

batch_size = 32

class MyDataset(Dataset):
    def __init__(self, data ):        
        self.df = data
    
    def __len__(self):
        return len( self.df )
    
    def data(self):
        return self.df

    def __getitem__( self, idx ):
        item = self.df.iloc[idx]
        return item['Seq'], item['Cmds']

datasets_wrapper = {}
for k in datasets.keys():
    datasets_wrapper[k] = (MyDataset( datasets[k][0] ), MyDataset( datasets[k][1] ) )
    
model_name = 't5-small'

tokenizer = T5Tokenizer( 'm.model', do_lower_case=True, do_basic_tokenize=True, 
                               padding=True, bos_token="<s>", 
                               eos_token="</s>",unk_token="<unk>", 
                               pad_token="<pad>")

SPECIAL_TOKENS_DICT = {
    'pad_token' : '<pad>',
    'additional_special_tokens' : ['<s>', '</s>']
}

tokenizer.add_special_tokens( SPECIAL_TOKENS_DICT ) 

def collate_fn( input ):

    src_input_ids = []
    trg_input_ids = []
    src_attention_masks = []
    trg_attention_masks = []

    max_len_seq = 0
    max_len_cmds = 0

    for inp in input:
        seq  = inp[0]
        cmds = inp[1]

        source_ids = tokenizer.encode_plus(seq)['input_ids']
        target_ids = tokenizer.encode_plus(cmds)['input_ids']

        if max_len_seq < len(source_ids):
            max_len_seq = len(source_ids)

        if max_len_cmds < len(target_ids):
            max_len_cmds = len(target_ids)

    for inp in input:
        seq  = inp[0]
        cmds = inp[1]

        source_dict = tokenizer.encode_plus(
                            seq + " </s>", # Sentence to encode.
                            max_length = max_len_seq,      # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            truncation = True,
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )

        target_dict = tokenizer.encode_plus(
                            cmds + " </s>", # Sentence to encode.
                            max_length = max_len_cmds,      # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            truncation = True,
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )

        src_input_ids.append( source_dict['input_ids'] )
        src_attention_masks.append( source_dict['attention_mask'] )

        trg_input_ids.append( target_dict['input_ids'] )
        trg_attention_masks.append( target_dict['attention_mask'] )


    src_input_ids = torch.cat(src_input_ids, dim=0 )    
    src_attention_masks = torch.cat(src_attention_masks, dim=0)

    trg_input_ids = torch.cat(trg_input_ids, dim=0 )    
    trg_attention_masks = torch.cat(trg_attention_masks, dim=0)

    #print(src_input_ids.shape, src_attention_masks.shape, trg_input_ids.shape, trg_attention_masks.shape)
    return ( src_input_ids, src_attention_masks, trg_input_ids, trg_attention_masks ) 


device = torch.device("cpu")
if torch.cuda.is_available():    
    device = torch.device("cuda")

epochs = 300
print( "Started training...")
best_acc = 0.0

torch.autograd.set_detect_anomaly(True)
K = 100
alpha = 0.01
beta = 1e-5 #0.001

config = T5Config(
    vocab_size = tokenizer.vocab_size,
    pad_token_id = tokenizer.pad_token_id,
    eos_token_id = tokenizer.eos_token_id,
    decoder_start_token_id = tokenizer.pad_token_id,
    #d_model = 300
)

def validate( tokenizer, model, device, loader ):

  model.eval()

  predictions = []
  actuals = []
  
  with torch.no_grad():
        for _, data in enumerate(loader, 0):
            src_ids = data[0].to(device)
            src_am  = data[1].to(device)
            trg_ids = data[2].to(device)

            generated_ids = model.generate(
                input_ids = src_ids,
                attention_mask = src_am, 
                max_length=60, 
                num_beams=2,
                #repetition_penalty=2.5, 
                #length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in trg_ids]
                        
            predictions.extend(preds)
            actuals.extend(target)

  return np.array( predictions ), np.array( actuals )

best_results = {}
num_samplings = 10

for k in datasets_wrapper.keys():
    k = 1
    
    model = T5ForConditionalGeneration( config ).to(device)
    model.resize_token_embeddings( len(tokenizer) )

    malm = l2l.algorithms.MAML(model, lr=alpha, first_order=False).to(device)
    optimizer = torch.optim.AdamW( malm.parameters(), lr=beta)
    adapt_steps = 5

    train, test = datasets_wrapper[k] 

    print(f'Started training for {k}')
    print( f'train.sz={len(train.data())} test.sz={len(test.data())}' )

    #test_loader = DataLoader( list(zip(test.data()['Seq'], test.data()['Cmds'])), collate_fn=collate_fn, batch_size=64)
    #train_loader = DataLoader( list(zip(train.data()['Seq'], train.data()['Cmds'])), collate_fn=collate_fn, batch_size=64)

    acc = 0.0

    for e in range(epochs):

        model.train()

        meta_train_loss = 0.0

        for _ in num_samplings:

            leaner = malm.clone()
            
            sample = train.data().sample(n=2*K)

            X_train = sample[:K]['Seq']
            y_train = sample[:K]['Cmds']

            X_test = sample[K:]['Seq']
            y_test = sample[K:]['Cmds']

            train_loader = DataLoader( list(zip(X_train, y_train)), collate_fn=collate_fn, batch_size=K)
            test_loader = DataLoader( list(zip(X_test, y_test)), collate_fn=collate_fn, batch_size=K)

            for train_batch in train_loader:
                        
                src_ids = train_batch[0].to(device)
                src_am  = train_batch[1].to(device)
                trg_ids = train_batch[2].to(device)

                lm_labels = trg_ids.clone().detach()
                lm_labels[trg_ids == tokenizer.pad_token_id] = -100
                
                for _ in range(adapt_steps):

                    outputs = learner(            
                        input_ids=src_ids,
                        attention_mask=src_am,
                        labels=lm_labels )

                    train_loss = outputs[0]

                    learner.adapt(train_loss)

                for test_batch in test_loader:
                        
                    src_ids = test_batch[0].to(device)
                    src_am  = test_batch[1].to(device)
                    trg_ids = test_batch[2].to(device)

                    lm_labels = trg_ids.clone().detach()
                    lm_labels[trg_ids == tokenizer.pad_token_id] = -100

                    outputs = learner(            
                        input_ids=src_ids,
                        attention_mask=src_am,
                        labels=lm_labels )

                    test_loss = outputs[0]

                    meta_train_loss += test_loss

            optimizer.zero_grad()
            meta_train_loss.backward()
            optimizer.step()

#            if e % 5 == 0:
            preds, actuals = validate( tokenizer, model, device, test_loader )
            prev_acc = acc
            acc = sum(preds == actuals) / len(actuals)

            if acc > prev_acc:
                best_results[k] = acc

            #print( '\tAcc updated!' )

            print( f'\t{k}:{e}: meta_train_loss: {meta_train_loss}, acc={acc}')
    
    print("\n============================\n")
    break

print( f'best_results={best_results}' )
