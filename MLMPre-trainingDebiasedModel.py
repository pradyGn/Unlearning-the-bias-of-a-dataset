import jsonlines
import transformers
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, BertModel
import torch
import pandas as pd
import os
import json
from tensorflow.keras.utils import to_categorical
import torch.nn as nn
import numpy as np
import random
from transformers import AdamW
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pickle
import re
from transformers import AdamW, get_linear_schedule_with_warmup

snli_train = []

with jsonlines.open('./snli_1.0/snli_1.0_train.jsonl') as f:
    for line in f.iter():
        s1 = line["sentence1"]
        s2 = line["sentence2"]
        label = line["gold_label"]
        snli_train.append([[s1, s2], label])

snli_dev = []

with jsonlines.open('./snli_1.0/snli_1.0_dev.jsonl') as f:
    for line in f.iter():
        s1 = line["sentence1"]
        s2 = line["sentence2"]
        label = line["gold_label"]
        snli_dev.append([[s1, s2], label])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



 class BertClassifier(nn.Module):
     def __init__(self, freeze_bert=False):
         super(BertClassifier, self).__init__()
         D_in, H, D_out = 30522, 32, 3
 
         self.bert = model
 
         self.classifier = nn.Sequential(
             nn.Dropout(0.1),
             nn.ReLU(),
             nn.Linear(D_in, H),
             nn.ReLU(),
             nn.Dropout(0.1),
             nn.Linear(H, D_out)
         )
 
         if freeze_bert:
             for param in self.bert.parameters():
                 param.requires_grad = False
         
     def forward(self, input_ids, attention_mask):
         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
         
         last_hidden_state_cls = outputs[0][:, 0, :]
 
         logits = self.classifier(last_hidden_state_cls)
 
         return logits

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

model.to(device)

def getTrainSentences(dataset):
  X_train = []
  y_train = []
  for i in range(len(dataset)):
    X_train.append(dataset[i][0][0] + " " + dataset[i][0][1])
    y_train.append(dataset[i][1])
  return X_train, y_train

def getsentence1(dataset):
  X_train = []
  y_train = []
  for i in range(len(dataset)):
    X_train.append(dataset[i][0][0])
    y_train.append(dataset[i][1])
  return X_train, y_train

snli_train_all, snli_y = getTrainSentences(snli_train)
snli_inputs = tokenizer(snli_train_all, return_tensors='pt', max_length=64, truncation=True, padding='max_length')
snli_inputs['labels'] = snli_inputs.input_ids.detach().clone()

#masking words only from premise to increase focus on premise
snli_train_premise, _ = getsentence1(snli_train)
snli_premise_inputs = tokenizer(snli_train_premise, return_tensors='pt', max_length=64, truncation=True, padding='max_length')

rand = torch.rand(snli_premise_inputs.input_ids.shape)
mask_arr = (rand < 0.15) * (snli_premise_inputs.input_ids != 101) * \
           (snli_premise_inputs.input_ids != 102) * (snli_premise_inputs.input_ids != 0)

selection = []

for i in range(snli_premise_inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

for i in range(snli_inputs.input_ids.shape[0]):
    snli_inputs.input_ids[i, selection[i]] = 103

class snliDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

snli = snliDataset(snli_inputs)

snliDataLoader = torch.utils.data.DataLoader(snli, batch_size=32, shuffle=True)

model.train()
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

def train(model, epochs, DataLoader, optim):
  for epoch in range(epochs):
      loop = tqdm(DataLoader, leave=True)
      for batch in loop:
          optim.zero_grad()
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs.loss
          loss.backward()
          optim.step()
          loop.set_description(f'Epoch {epoch}')
          loop.set_postfix(loss=loss.item())

train(model, 2, snliDataLoader, optim)

torch.save(model, "./MLMPre-trainedDebiasedModel.pth")
