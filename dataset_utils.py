import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import _stateless
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm.notebook import tqdm
import pickle


MAX_LENGTH = 256


class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        self.max_length = 512
        self.dataset = dataset
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids, labels = self.dataset[idx]
        input_ids = torch.tensor(input_ids + [0] * (self.max_length - len(input_ids))).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        return input_ids, labels


def load_nlp_dataset(task, subset_size=-1):
    if task == 'machine-translation':
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
        dataset = load_dataset("opus_books", "en-fr")
        dataset = dataset['train'].shuffle()
        if subset_size != -1:
            dataset = dataset[:subset_size]
        prefix = "translate English to French: "
        input_ids = [tokenizer.encode(prefix + row['en'])[:MAX_LENGTH] for row in tqdm(dataset['translation'])]
        with tokenizer.as_target_tokenizer():
            labels = [tokenizer.encode(row['fr'])[:MAX_LENGTH] for row in tqdm(dataset['translation'])]
        dataset = list(zip(input_ids, labels))
    elif task == 'binary-classification':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        dataset = load_dataset("imdb")
        dataset = dataset['train'].shuffle()
        if subset_size != -1:
            dataset = dataset[:subset_size]
        dataset = [(tokenizer.encode(row['text'], truncation=True), row['label']) for row in tqdm(dataset)]

    torch.save(dataset, 'data/train_dataset.pth')


def load_nlp_dataloader(batch_size):
    dataset = torch.load('data/train_dataset.pth')
    _dataset = NLPDataset(dataset, 'cuda')
    dataloader = torch.utils.data.DataLoader(_dataset, batch_size=batch_size, shuffle=True)
    torch.save(dataloader, 'data/train_dataloader.pth')


def load_model(task):
    if task == 'machine-translation':
        model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
        model = model.to('cuda')
    elif task == 'binary-classification':
        model = BertForSequenceClassification.from_pretrained('bert-base-cased')
        model = model.to('cuda')

    dataloader = torch.load('data/train_dataloader.pth')
    return model, dataloader

