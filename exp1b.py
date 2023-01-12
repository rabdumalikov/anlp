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
    
model_name = 'xlm-roberta-base'

model_name = 't5-small'

tokenizer = T5Tokenizer.from_pretrained( model_name, do_lower_case=True, model_max_length=512)

SPECIAL_TOKENS_DICT = {
    'pad_token' : '<pad>',
    'additional_special_tokens' : ['<question>', '<context>']
}

tokenizer.add_special_tokens( SPECIAL_TOKENS_DICT ) 

def collate_fn( input ):

    src_input_ids = []
    trg_input_ids = []
    src_attention_masks = []
    trg_attention_masks = []

    max_len = 9

    for inp in input:
        seq  = inp[0]
        cmds = inp[1]
    
        source_dict = tokenizer.encode_plus(
                            [seq], # Sentence to encode.
                            max_length = max_len,      # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            truncation = True,
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )

        target_dict = tokenizer.encode_plus(
                            [cmds], # Sentence to encode.
                            max_length = 48,      # Pad & truncate all sentences.
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

    return TensorDataset( src_input_ids, src_attention_masks, trg_input_ids, trg_attention_masks ) 


device = torch.device("cpu")
if torch.cuda.is_available():    
    device = torch.device("cuda")

epochs = 2
print( "Started training...")
best_acc = 0.0

torch.autograd.set_detect_anomaly(True)
K = 10
alpha = 0.01
beta = 0.001

train, test = datasets_wrapper[1] 

model = T5ForConditionalGeneration.from_pretrained( model_name ).to(device)
model.resize_token_embeddings( len(tokenizer) )

#print(dir(model))
optimizer = torch.optim.AdamW( model.parameters(), lr=1e-5, eps=1e-8)
weights = list(model.parameters())
loss_fn = nn.CrossEntropyLoss()

for e in range(epochs):

    sample = train.data().sample(n=2*K)

    X_train = sample[:K]['Seq']
    y_train = sample[:K]['Cmds']

    X_test = sample[K:]['Seq']
    y_test = sample[K:]['Cmds']

    dl = DataLoader( list(zip(X_train, y_train)), collate_fn=collate_fn, batch_size=1)
    
    meta_train_loss = 0.0

    for tens in dl:
        src_ids = tens[0][0].unsqueeze(dim=0).to(device)
        src_am  = tens[0][1].unsqueeze(dim=0).to(device)
        trg_ids = tens[0][2].unsqueeze(dim=0).to(device)
        trg_am  = tens[0][3].unsqueeze(dim=0).to(device)

        y_ids = trg_ids[:,:-1].contiguous()
        lm_labels = trg_ids[:,1:].clone().detach()
        lm_labels[trg_ids[:,1:] == tokenizer.pad_token_id] = -100

        print( src_ids.shape, src_am.shape, y_ids.shape, lm_labels.shape )

        temp_weights = [w.clone() for w in weights]

        outputs = model(
            input_ids=src_ids,
            attention_mask=src_am,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )

        train_loss = outputs[0]

        grad = torch.autograd.grad(train_loss, weights)
        temp_weights = [ (w - alpha * g).tolist() for w, g in zip(weights, grad)]

        print( nn.Parameter( temp_weights ) )
        exit()

        model.load_state_dict( dict(zip(model.state_dict().keys(), temp_weights)) )

        test_loss = loss_fn(y_test, model(X_test))

        meta_train_loss += test_loss

    optimizer.zero_grad()
    meta_train_loss.backward()
    optimizer.step()

    print( f'meta_train_loss: {meta_train_loss}' )


exit()

for epoch in range(epochs):
    
    model.train()
    losses = []
    
    for step, batch in enumerate(train_dataloader):

        keys = batch[-1]
    
        for k in keys:
        
            numbatches = batch[0][k].size()[0]

            for b in range(0, numbatches, batch_size ):

                b_input_ids = batch[0][k][b:b+batch_size].to(device)
                b_tt_ids = batch[1][k][b:b+batch_size].to(device)
                b_am = batch[2][k][b:b+batch_size].to(device)
                b_lbl = batch[3][k][b:b+batch_size].to(device)

                optimizer.zero_grad()
                outputs = model( b_input_ids, token_type_ids=b_tt_ids, attention_mask=b_am, labels=b_lbl )
                
                loss, logits = outputs[:2]

                loss.backward()

                optimizer.step()

                losses.append( loss.item() ) #torch.mean( torch.tensor( [loss1.item(), loss2.item(), loss3.item()] ) ) )

    ppl = evaluate( val_dataloader, model )
    
    print(f'Validation accuracy at {epoch}: ppl={ppl} train loss: {sum(losses) / len(losses)}')

    print( "=" * 20 )

    model.eval()

    for i in things_to_ask:
        input_ids = tokenizer.encode( i, add_special_tokens=True, return_tensors='pt')
        input_ids = input_ids.to(device)

        # generate text until the output length (which includes the context length) reaches 50
        greedy_output = model.generate(input_ids, top_p=0.95, do_sample=True, max_length=50, pad_token_id=tokenizer.eos_token_id )

        print( tokenizer.decode(greedy_output[0], skip_special_tokens=True))

    # Keep track of the best model based on the accuracy
    model.save_pretrained("best_eng_model_GPT2_window_data_all_data")

print( "Finished training.")



exit()

train_dataloader = DataLoader( train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False )

val_dataloader_eng = DataLoader( val_subdataset_eng, collate_fn=collate_fn, batch_size=batch_size, shuffle=False )
val_dataloader_fin = DataLoader( val_subdataset_fin, collate_fn=collate_fn, batch_size=batch_size, shuffle=False )
val_dataloader_jap = DataLoader( val_subdataset_jap, collate_fn=collate_fn, batch_size=batch_size, shuffle=False )

import numpy as np
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import mean_squared_error


bert_hidden_dim_size = 768
hidden_dim_size = 512

class myModel(nn.Module):
    def __init__( self ):

        super().__init__()

        self.model = GPT2LMHeadModel.from_pretrained( model_name )
        self.model = AutoModel.from_pretrained( model_name )

        # heads
        self.iob_head = nn.Linear( bert_hidden_dim_size, 3 )
        self.binary = nn.Linear( bert_hidden_dim_size, 2 )

    def forward( self, input_ids, token_type_ids, attention_mask, labels = None ):

        trans_out = self.model( input_ids, attention_mask=attention_mask )
          
        batch_logits = trans_out['last_hidden_state']

        iob_output = self.iob_head( batch_logits ) 
        bin_output = self.binary( batch_logits ) 

        return iob_output, bin_output

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

def accuracy( pred, target ):
    pred = np.asarray(pred)
    target = np.asarray(target)

    #print( f'pred={pred} trg={target}' )

    P, R, F1, _ = precision_recall_fscore_support(target, pred, average='macro')

    return ( np.sum( pred == target ) / float(target.shape[0]) ), F1

def evaluate( valid_df, model ):

    model.eval()

    label_iob_all = []
    output_iob_all = []
    label_cls_all = []
    output_cls_all = []

    total_mse = 0.0

    for batch in valid_df:
        b_input_ids = batch[:][0].to(device)
        b_tt = batch[:][1].to(device)
        b_am = batch[:][2].to(device)
        b_lbl = batch[:][3]
        b_ans_lbl = batch[:][4]
        
        with torch.no_grad():  
            iob_distr, bin_distr = model( b_input_ids, b_tt, attention_mask=b_am )
            iob_distr = torch.argmax( iob_distr, axis=-1 )
            iob_distr = iob_distr.view(-1)

            bin_distr = torch.argmax( torch.sum(bin_distr, dim=1), axis=-1 )
            #print( f'bin_distr={bin_distr.size()} b_ans_lbl={b_ans_lbl.size()}' )

            bin_distr = bin_distr.view(-1)

            idxs = np.where( b_lbl.view(-1).numpy() != -100 )[0]

            output_iob_all.extend( list( iob_distr[idxs].detach().clone().cpu().numpy()) )
            label_iob_all.extend( list( b_lbl.view(-1)[idxs] ) )

            output_cls_all.extend( list( bin_distr.detach().clone().cpu().numpy()) )
            label_cls_all.extend( list( b_ans_lbl.view(-1) ) )

    return accuracy( output_iob_all, label_iob_all ), accuracy( output_cls_all, label_cls_all )


device = torch.device("cpu")
if torch.cuda.is_available():    
    device = torch.device("cuda")

from transformers import TrainingArguments, Trainer

model = myModel().to(device)
optimizer = torch.optim.AdamW( model.parameters(), lr=1e-5, eps=1e-8)

iss = 0 # 0
oss = 0 # 1
bss = 0 # 2

for step, batch in enumerate(train_dataloader):
    b_lbl = batch[:][3]
    arr = b_lbl.view(-1).numpy()
    
    iss += np.where( arr == 0 )[0].shape[0]
    oss += np.where( arr == 1 )[0].shape[0]
    bss += np.where( arr == 2 )[0].shape[0]

mss = max( [iss, oss, bss ] )

print( f'mss={mss} iss={iss} oss={oss} bss={bss}' )

weight = torch.tensor( [mss/iss, mss/oss, mss/bss] ).to(device)

print( f'weight={weight}')

loss_fn = nn.CrossEntropyLoss( weight=weight )
loss_fn2 = nn.CrossEntropyLoss()

epochs = 200
print( "Started training...")
best_acc = 0.0

for epoch in range(epochs):
    
    model.train()
    losses = []
    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[:][0].to(device)
        b_tt = batch[:][1].to(device)
        b_am = batch[:][2].to(device)
        b_lbl = batch[:][3].to(device)
        b_ans_lbl = batch[:][4].to(device)

        optimizer.zero_grad()

        iob_distr, bin_distr = model( b_input_ids, b_tt, attention_mask=b_am )

        bin_distr = torch.sum( bin_distr, dim=1 )
        #print( f'iob_distr={iob_distr.size()} bin_distr={bin_distr.size()}' )

        iob_distr = iob_distr.view( -1, 3 )
        bin_distr = bin_distr.view( -1, 2 )
        
        b_lbl = b_lbl.view( -1 )

        #print( f'bin_distr={bin_distr.size()} b_ans_lbl={b_ans_lbl.size()}' )
        loss = loss_fn( iob_distr, b_lbl )
        loss += loss_fn2( bin_distr, b_ans_lbl )

        loss.backward()
        optimizer.step()

        losses.append( loss.item() ) #torch.mean( torch.tensor( [loss1.item(), loss2.item(), loss3.item()] ) ) )

    names = [ 'ENG', 'FIN', 'JPN' ]
    for ii, val_dataloader in enumerate( [val_dataloader_eng, val_dataloader_fin, val_dataloader_jap] ):
        iob, clas = evaluate( val_dataloader, model )
        print(f'{names[ii]}: Validation accuracy at {epoch}: acc_cls={clas[0]} f1scr_cls={clas[1]} acc_iob={iob[0]} f1scr_iob={iob[1]} train loss: {sum(losses) / len(losses)}')

    # Keep track of the best model based on the accuracy
    if clas[0]+iob[0] > best_acc:
      torch.save(model.state_dict(), f'l6_roberta.pt')
      best_acc = clas[0]+iob[0]


print( "Finished training.")