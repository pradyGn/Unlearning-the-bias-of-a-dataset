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

snli_train = []

with jsonlines.open('/content/drive/MyDrive/DLSProject_0514/snli_1.0/snli_1.0_train.jsonl') as f:
    for line in f.iter():
        s1 = line["sentence1"]
        s2 = line["sentence2"]
        label = line["gold_label"]
        snli_train.append([[s1, s2], label])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

model.to(device)

def getsentence2(dataset):
  X_train = []
  y_train = []
  for i in range(len(dataset)):
    X_train.append(dataset[i][0][1])
    y_train.append(dataset[i][1])
  return X_train, y_train

snli_train, snli_y = getsentence2(snli_train)

snli_inputs = tokenizer(snli_train, return_tensors='pt', max_length=64, truncation=True, padding='max_length')

snli_inputs.keys()

snli_inputs['labels'] = snli_inputs.input_ids.detach().clone()

snli_inputs.keys()

rand = torch.rand(snli_inputs.input_ids.shape)
mask_arr = (rand < 0.15) * (snli_inputs.input_ids != 101) * \
           (snli_inputs.input_ids != 102) * (snli_inputs.input_ids != 0)

mask_arr

selection = []

for i in range(snli_inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

for i in range(snli_inputs.input_ids.shape[0]):
    snli_inputs.input_ids[i, selection[i]] = 103

snli_inputs.input_ids

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