import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold

import torch
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *


class telecomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
def eval_step(engine, batch):
    return batch


class neural_net(torch.nn.Module):
    def __init__(self, intput_dim, _batch_size=32):
        super(neural_net, self).__init__()
        self.model = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(intput_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 1024),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.BatchNorm1d(1024),

            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.ReLU(),

            nn.BatchNorm1d(256),

            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),

            nn.Sigmoid()
            # PyTorch implementation of cross-entropy loss includes softmax layer
        )
        self.loss = nn.BCELoss()
        self.default_evaluator = Engine(eval_step)
        roc_auc = ROC_AUC()
        roc_auc.attach(self.default_evaluator, 'roc_auc')
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = _batch_size

    def train(self, X_train, Y_train, n_epochs=100, train_all =  False):
        N_train = int(X_train.shape[0] * 4 / 5)

        selected_item = np.random.choice(np.arange(X_train.shape[0]), N_train, replace=False)

        print('Number of training data is ',N_train)
        X_train_train = X_train[selected_item,:]
        X_train_val = X_train[~np.isin(np.arange(X_train.shape[0]), selected_item),:]
        Y_train_train = Y_train[selected_item]
        Y_train_val = Y_train[~np.isin(np.arange(X_train.shape[0]), selected_item)]

        print('Shape of Y_train_train is ',Y_train_train.shape)

        # create training dataset
        self.train_dataset = telecomDataset(torch.tensor(X_train_train).float(), torch.tensor(Y_train_train).float())
        self.test_dataset = telecomDataset(torch.tensor(X_train_val).float(), torch.tensor(Y_train_val).float())

        if (train_all):
            self.train_dataset = telecomDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
            self.test_dataset = telecomDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)        
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        training_accuracy_history = np.zeros([n_epochs, 1])
        training_loss_history = np.zeros([n_epochs, 1])
        validation_accuracy_history = np.zeros([n_epochs, 1])
        validation_loss_history = np.zeros([n_epochs, 1])

        for epoch in range(n_epochs):
            print(f'Epoch {epoch+1}/{n_epochs}:', end='')
            train_total = 0
            # 
            acc = 0
            self.model.train()
            for i, data in enumerate(self.train_loader):
                images, labels = data
                self.optimizer.zero_grad()
                # forward pass
                output = self.model(images)
                # calculate categorical cross entropy loss
                loss = self.loss(output, labels)
                # backward pass
                loss.backward()
                self.optimizer.step()
                
                # track training accuracy
                #output = (output > 0.5).float()
                #train_total += labels.size(0)
                #train_correct += (output == labels).sum().item()
                # track training loss
                training_loss_history[epoch] += loss.item()
                # progress update after 180 batches (~1/10 epoch for batch size 32)
                if i % 180 == 0: print('.',end='')
                try:
                    State = self.default_evaluator.run([[output, labels]])
                    if (State == None):
                        continue
                    acc += State.metrics['roc_auc']*labels.size(0)
                    train_total += labels.size(0)
                except ValueError:
                    pass
                
                
                

            training_loss_history[epoch] /= len(self.train_loader)
            if train_total == 0:
                training_accuracy_history[epoch] = 0
            else:
                training_accuracy_history[epoch] = acc / train_total
            print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}, ROC AUC: {training_accuracy_history[epoch,0]:0.4f}',end='')
                
            # validate
            test_total = 0
            test_acc = 0
            with torch.no_grad():
                self.model.eval()
                for i, data in enumerate(self.test_loader):
                    images, labels = data
                    # forward pass
                    output = self.model(images)
                    # find accuracy
                    #output = (output > 0.5).float()
                    #test_total += labels.size(0)
                    #test_correct += (output == labels).sum().item()
                    # find loss
                    loss = self.loss(output, labels)
                    validation_loss_history[epoch] += loss.item()
                    State = self.default_evaluator.run([[output, labels]])
                    if (State == None):
                        continue
                    test_acc += State.metrics['roc_auc']*labels.size(0)
                    test_total += labels.size(0)

                validation_loss_history[epoch] /= len(self.test_loader)
                if test_total == 0:
                    validation_accuracy_history[epoch] = 0
                else:
                    validation_accuracy_history[epoch] = test_acc / test_total
            print(f', val loss: {validation_loss_history[epoch,0]:0.4f}, ROC AUC: {validation_accuracy_history[epoch,0]:0.4f}')

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X).float()).detach().numpy()