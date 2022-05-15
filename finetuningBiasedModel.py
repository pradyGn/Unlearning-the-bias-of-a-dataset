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
import pickle
import re
from transformers import AdamW, get_linear_schedule_with_warmup
import time

snli_train = []

with jsonlines.open('./snli_1.0/snli_1.0_train.jsonl') as f: #add appropriate path to datasets
    for line in f.iter():
        s1 = line["sentence1"]
        s2 = line["sentence2"]
        label = line["gold_label"]
        snli_train.append([[s1, s2], label])

snli_dev = []

with jsonlines.open('./snli_1.0/snli_1.0_dev.jsonl') as f: #add appropriate path to datasets
    for line in f.iter():
        s1 = line["sentence1"]
        s2 = line["sentence2"]
        label = line["gold_label"]
        snli_dev.append([[s1, s2], label])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = torch.load("./MLMPre-trainedBiasedModel.pth") #add path to MLM Pre-trained Biased model.

model.to(device)

def getsentence2(dataset):
  X_train = []
  y_train = []
  for i in range(len(dataset)):
    X_train.append(dataset[i][0][1])
    y_train.append(dataset[i][1])
  return X_train, y_train

snli_train, snli_y = getsentence2(snli_train)

class snliDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


snli_train_X = snli_train
snli_train_y = snli_y
snli_dev_X, snli_dev_y = getsentence2(snli_dev)

snli_train_inputs = tokenizer(snli_train_X, max_length=64, truncation=True, padding='max_length')
snli_dev_inputs = tokenizer(snli_dev_X, max_length=64, truncation=True, padding='max_length')

snli_train_inputs.input_ids = torch.Tensor(snli_train_inputs.input_ids).long()
snli_train_inputs.token_type_ids = torch.Tensor(snli_train_inputs.token_type_ids).long()
snli_train_inputs.attention_mask = torch.Tensor(snli_train_inputs.attention_mask).long()

snli_dev_inputs.input_ids = torch.Tensor(snli_dev_inputs.input_ids).long()
snli_dev_inputs.token_type_ids = torch.Tensor(snli_dev_inputs.token_type_ids).long()
snli_dev_inputs.attention_mask = torch.Tensor(snli_dev_inputs.attention_mask).long()

def numerize_y(y):
  numed_y = []
  for label in y:
    if label == "contradiction":
      numed_y.append(0)
    elif label == "entailment":
      numed_y.append(1)
    else:
      numed_y.append(2)
  return numed_y

snli_train_y = numerize_y(snli_train_y)
snli_dev_y = numerize_y(snli_dev_y)

Trainlabel_tensor = torch.Tensor(snli_train_y).long()
Testlabel_tensor = torch.Tensor(snli_dev_y).long()

snli_train_inputs['labels'] = Trainlabel_tensor
snli_dev_inputs['labels'] = Testlabel_tensor

snli_train_data = TensorDataset(snli_train_inputs.input_ids, snli_train_inputs.attention_mask, snli_train_inputs.labels)
train_sampler = RandomSampler(snli_train_data)
snli_train_dataloader = DataLoader(snli_train_data, sampler=train_sampler, batch_size=32)

snli_dev_data = TensorDataset(snli_dev_inputs.input_ids, snli_dev_inputs.attention_mask, snli_dev_inputs.labels)
dev_sampler = RandomSampler(snli_dev_data)
snli_dev_dataloader = DataLoader(snli_dev_data, sampler=dev_sampler, batch_size=32)


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

def initialize_model(epochs=4):
    bert_classifier = BertClassifier(freeze_bert=False)

    bert_classifier.to(device)

    optimizer = AdamW(bert_classifier.parameters(),
                      lr=2e-5,    
                      eps=1e-8
                      )

    total_steps = len(snli_train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    print("Start training...\n")
    for epoch_i in range(epochs):
        # Training
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1

            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(b_input_ids, b_attn_mask)

            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # Evaluation
        if evaluation == True:
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    predictions = []

    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        predictions.append(preds)

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

set_seed(42)
biased_classifier, optimizer, scheduler = initialize_model(epochs=3)
train(biased_classifier, snli_train_dataloader, snli_dev_dataloader, epochs=6)

torch.save(biased_classifier, "./finetunedBiasedModel.pth")

