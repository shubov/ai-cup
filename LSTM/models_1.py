from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

### Simple Fully Connected FeedForward Neural Network
class LSTM(nn.Module):
    def __init__(self,
                 input_size: int):
        """
        LSTM model
        Args
        input_size: dimensions in the feature space.
        """
        super(LSTM, self).__init__()
        self.hidden_layers = input_size
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)
        
    def forward(self, x, future_preds=0):
        outputs, num_samples = [], x.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        for time_step in x.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(input_t, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)
            
        for i in range(future_preds):
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
            # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        
        return outputs





class SeasonalPredictor(BaseEstimator):
    def __init__(self,
                 gap: int = 7 ,
                 date_col:str = 'date'):
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

