from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Simple Fully Connected FeedForward Neural Network
class MLP(nn.Module):
    def __init__(self,
                 input_size: int):
        """
        Simple 4 layer FC network.
        Args
        input_size: dimensions in the feature space.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x : (batch_size, n, input_size):
        out = self.activation(self.fc3(self.fc2(self.fc1(self.fc(x)))))
        # out : (batch_size, n, 1)
        return out


class SeasonalPredictor(BaseEstimator):
    def __init__(self,
                 gap: int = 7,
                 date_col: str = 'date'):
        self.gap = gap
        self.date_col = date_col

    def fit(self, X, y):
        self.dates_ = X[self.date_col]
        self.y_ = np.array(y)
        return self

    def predict(self, X):
        pred_dates = (X[self.date_col] - pd.to_timedelta(self.gap, unit='d')).values
        return self.y_[np.where(np.isin(self.dates_, pred_dates))[0]]


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
            y_pred = model(seq)
            loss = loss_function(y_pred.squeeze(), labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        # print every epoch
        if i % 1 == 0:
            print(f'epoch: {i:3} loss: {epoch_loss:10.8f}')

    # save weights
    torch.save(model.state_dict(), "model.h5")
    print("Finished training and model saved!")
