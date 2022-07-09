from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    """
    LSTM implementation

    """

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.5)
        #self.fc1 = nn.Linear(hidden_size, 32)
        #self.fc2 = nn.Linear(32, 16)
        #self.fc3 = nn.Linear(16, 1)
        #self.activation = nn.ReLU()
        self.fc=nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()




    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, 61, self.hidden_size).requires_grad_()
        c_0 = torch.zeros(self.num_layers, 61, self.hidden_size).requires_grad_()
        # Propagate input through LSTM
        out, hn = self.lstm(x, (h_0, c_0))
        #out_2, hn_2 = self.lstm(hn, (h_2, c_2))

        #output=self.activation(self.fc3(self.fc2(self.fc1(out))))
        output=self.activation(self.fc(out))
        return output




def train_model(model, data, device, **params):
    """Simple training loop for a pytorch model"""
    # loss function
    model.to(device)
    print(model)
    loss_function = nn.MSELoss()
    # optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # main training loop
    for i in range(params['num_epochs']):
        epoch_loss = 0
        for j, sample in enumerate(data):
            seq = sample[0].float().to(device)
            labels = sample[1].float().to(device)
            optimizer.zero_grad()
            out=model(seq)
            loss = loss_function(out.squeeze(), labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        # print every epoch
        if i % 1 == 0:
            print(f'epoch: {i:3} loss: {epoch_loss:10.8f}')

    # save weights
    torch.save(model.state_dict(), "model_2.h5")
    print("Finished training and model saved!")

