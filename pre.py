# import stuff

import torch
import torch.nn as nn
import numpy as np

import yfinance as yf

from collections import OrderedDict
from os import path

import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
take = 16
ticker = ["MSFT", "AAPL", "T", "WMT", "AMZN", "XOM", "CVX", "JPM",
                  "VZ", "CMcSA", "WFC", "C", "TGT", "IBM", "BA", "INTC", "FDX"]
data_x, data_y, val_x, val_y, close = {}, {}, {}, {}, {}
if not (path.exists('data/pre/data_x.npy')):
    for tick in ticker:
        close[tick] = pd.read_csv('data/perc/'+tick+'.csv',  index_col=0)
        l = len(close[tick]['Close'])
        val_indices = close['MSFT'].index[0:int(0.2*l)]

        data_x[tick] = [close[tick]['Close'][i: i+take].to_numpy()
                        for i in range(l-take) if close[tick].index[i+take] not in val_indices]
        val_x[tick] = [close[tick]['Close'][i: i+take].to_numpy()
                       for i in range(l-take) if close[tick].index[i+take] in val_indices]

        data_y[tick] = [[close[tick]['Close'][i+take]]
                        for i in range(l-take) if close[tick].index[i+take] not in val_indices]
        val_y[tick] = [[close[tick]['Close'][i+take]]
                       for i in range(l-take) if close[tick].index[i+take] in val_indices]

    data_x = torch.FloatTensor(np.concatenate(
        [data_x[tick] for tick in ticker])).float().reshape(-1, take)[:]
    data_y = torch.FloatTensor(np.concatenate(
        [data_y[tick] for tick in ticker])).float().reshape(-1, 1)[:]
    val_x = torch.FloatTensor(np.concatenate(
        [val_x[tick] for tick in ticker])).float().reshape(-1, take)[:].to(device)
    val_y = torch.FloatTensor(np.concatenate(
        [val_y[tick] for tick in ticker])).float().reshape(-1, 1)[:].to(device)
    torch.save(data_x, 'data/pre/data_x.npy')
    torch.save(data_y, 'data/pre/data_y.npy')
    torch.save(val_x, 'data/pre/val_x.npy')
    torch.save(val_y, 'data/pre/val_y.npy')

data_x = torch.load('data/pre/data_x.npy')
data_y = torch.load('data/pre/data_y.npy')
val_x = torch.load('data/pre/val_x.npy')
val_y = torch.load('data/pre/val_y.npy')


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('Blin1', nn.Linear(take, take)),
            ('Belu1', nn.ELU()),
            ('Blin2', nn.Linear(take, take)),
            ('Belu2', nn.ELU()),
            ('Blin3', nn.Linear(take, 1)),
            # ('Bsigm2', nn.Tanh())
        ]))

    def forward(self, x):
        x = self.layers(x)
        return x


net = SimpleModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, betas=(
    0.5, 0.8), eps=0.000000001, weight_decay=0.1)
batchsize = 512
epoch = 1000
amount = data_y.shape[0]
amount_val = val_y.shape[0]
cnt = int(amount / batchsize)+1
cnt_val = int(amount_val / batchsize)+1

print('\r\n\r\n\r\n----------------------------------------------------------------------------------------')
print('start training on previous {} days'.format(take))
print('variance on train data is : {:3.2f} - on validation data is : {:3.2f} - train always buy : {:3.1f} - val always buy : {:3.1f}'.format(
    data_y.var(), val_y.var(), int(1000 * (sum(data_y >= 0.0)).item() / data_y.shape[0]) / 10, int(1000 * (sum(val_y >= 0.0)).item() / val_y.shape[0]) / 10))
loss_train = []
loss_val = []
majority = torch.zeros(val_y.shape[0]).to(device)

for e in range(epoch):

    if (e % 1 == 0):
        net.eval()
        strades = correct = scorrect = buys = mcorrect = 0
        with torch.no_grad():
            strong = 3

            outB = net(val_x)
            cVal = criterion(outB, val_y)
            if (e % 10 == 0):
                strades += sum(outB > 0.0 + strong) + \
                    sum(outB < 0.0 - strong)
                correct += sum(torch.logical_and(outB >= 0.0, val_y >= 0.0)) + \
                    sum(torch.logical_and(outB <= 0.0, val_y <= 0.0))
                scorrect += sum(torch.logical_and(outB >= 0.0 + strong, val_y >= 0.0)) + \
                    sum(torch.logical_and(outB <= 0.0 - strong, val_y <= 0.0))
                buys += sum(outB > 0)
                majority += torch.sign(outB.reshape(-1))
                mcorrect += sum(torch.logical_and(majority > 0, val_y.reshape(-1) > 0)) + \
                    sum(torch.logical_and(majority <= 0, val_y.reshape(-1) <= 0))
                if (strades == 0):
                    strades += 1
                print('Epoch: {:3}  - train loss : {:3.4f} - val loss: {:3.4f} trades: {} - buys : {} - all val: {:3.2f} strades: {}- strong val: {:3.2f} - mcorrect : {}%'.format(e, torch.mean(torch.FloatTensor(loss_train)), cVal, val_y.shape[0], buys.item(),
                                                                                                                                                                                   int(1000*correct / val_y.shape[0])/10,   strades.item(), int(1000*scorrect / strades)/10, int(1000*mcorrect / val_y.shape[0])/10))
            else:
                print('Epoch: {:3}  - train loss : {:3.4f} - val loss: {:3.4f}'.format(
                    e, torch.mean(torch.FloatTensor(loss_train)), cVal))
    # torch.cuda.empty_cache()
    net.train()
    for j in range(cnt):

        bx = data_x[batchsize*j:batchsize*(j+1)].to(device)
        by = data_y[batchsize*j:batchsize*(j+1)].to(device)
        optimizer.zero_grad()
        output = net(bx)
        loss = criterion(output, by)
        loss_train.append(loss)
        if not loss > 0:
            print('woops')
        if (j % 10 == 0):
            print('batch loss : {:3.4f}'.format(loss.item()), end='\r')
        loss.backward()
        optimizer.step()
