import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, concatenate_datasets
from collections import defaultdict
import random
from samplers import BalancedSampler, UnbalancedSampler
from models import BinaryClassifier, CrossEncoder, CE_PA
from utils import batch_from_episode, evaluate
from tqdm import tqdm

model_name = 'shiva-avihs/intendd-roberta-for-pretraining'
dataset_name = 'clinc150'
num_classes = 150
mode = 'val'
sampling_mode = 'balanced'

#Load Model
if model_name == 'shiva-avihs/intendd-roberta-for-pretraining':
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = CE_PA(model_name)
model.to(device)

#Load Data
if dataset_name == 'banking77':
    dataset = load_dataset('PolyAI/banking77')
    train_data = dataset['train']
    test_data = dataset['test']
    dataset = concatenate_datasets([train_data, test_data])
    label_tag = 'label'
    
elif dataset_name == 'clinc150':
    dataset = load_dataset('clinc_oos','small')
    train_data = dataset['train']
    test_data = dataset['test']
    val_data = dataset['validation']
    dataset = concatenate_datasets([train_data, test_data])
    dataset = concatenate_datasets([dataset, val_data])
    label_tag = 'intent'

labels_list = list(set(dataset[label_tag]))
full_dataset = defaultdict(list)
print("Preproccessing dataset")
for n in tqdm(range(len(dataset))):
        text = dataset['text'][n]
        label = dataset[label_tag][n]
        full_dataset[label].append(text)
full_dataset = dict(full_dataset)


if dataset_name == 'banking77':
    train_len = 25
    val_len = 25
    test_len = 27
elif dataset_name == 'clinc150':
    train_len = 50
    val_len = 50
    test_len = 50

class_list = list(full_dataset.keys())
random.seed(1234)
random.shuffle(class_list)
train_classes = class_list[:train_len]
val_classes = class_list[train_len:train_len+val_len]
test_classes = class_list[train_len+val_len:]

if sampling_mode == 'balanced':
    Sampler = BalancedSampler
    
elif sampling_mode == 'unbalanced':
    Sampler = UnbalancedSampler

n_way = 5
k_shot = 5

train_sampler = Sampler(full_dataset, train_classes, n_way, k_shot)
if mode == 'val': 
    eval_sampler = Sampler(full_dataset, val_classes, n_way, k_shot)
elif mode == 'test': 
    eval_sampler = Sampler(full_dataset, test_classes, n_way, k_shot)

#Train
train_episodes = 100000
eval_episodes = 10
learning_rate = 2e-5
batch_size = 64 #Should not affect training results implemented for memory efficiency

loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_accuracies = []
eval_accuracies = []

print("Beginning Training")
for ep_idx in tqdm(range(train_episodes)):
    model.train()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    episode = train_sampler.sample()
    for batch in batch_from_episode(episode, batch_size):
        x_support = batch[0]
        x_query = batch[1]
        y = batch[2].to(device)
        scores = model(x_support,x_query)
        batch_loss = nn.functional.binary_cross_entropy(scores.squeeze(1), y)
        batch_loss.backward()
    optimizer.step()

    if ep_idx%100 == 99:
        train_accuracies.append(evaluate(model, eval_episodes, batch_size, train_sampler))
        print('train: ',train_accuracies[-1])
        eval_accuracies.append(evaluate(model, eval_episodes, batch_size, eval_sampler))
        print('eval: ',eval_accuracies[-1])

print("Evaluating final model")
accuracy = evaluate(model, 600, batch_size, eval_sampler)
print('Final eval accuracy = ',accuracy)
