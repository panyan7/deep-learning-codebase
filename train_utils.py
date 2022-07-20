import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pickle
import collections
import gc
import resource
import sys
import os
import random


TEST_EPOCHS = [1, 2, 3, 8, 9, 10]
NUM_SAMPLE = 50


def train(model, dataloader, num_epochs, grad_accumulation, optimizer, num_epoch_to_save=1):
    loss_list = []
    for epoch in range(num_epochs):
        loss = 0.0
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids, labels = batch
            out = model(input_ids=input_ids, labels=labels)
            loss1 = out[0]
            loss += float(loss1) / len(dataloader)
            loss1 /= grad_accumulation

            if i % grad_accumulation == 0:
                batch_loss = 0.0

            loss1.backward(retain_graph=True)

            batch_loss += float(loss1)

            if (i+1) % grad_accumulation == 0 or i+1 == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

                del batch_data

                torch.cuda.empty_cache()

            del input_ids, labels, out

            torch.cuda.empty_cache()

        loss_list.append(loss)

        print(f'Epoch {epoch+1}, loss {loss}')

        if (epoch+1) % num_epoch_to_save == 0:
            torch.save(model, f'model/model.pt')

    del optimizer

    return loss_list


