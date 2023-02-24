# import stuff
if 1 > 0:
    import torch
    import torch.nn as nn
    import numpy as np

    import yfinance as yf

    from collections import OrderedDict
    from os import path

    import pandas as pd

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvCrusherNetPartA(nn.Module):
    def __init__(self, channels, input_size, kernelsizes, dropout=0.5):
        super().__init__()
        self.channels = channels
        self.kernelsizes = kernelsizes
        self.layers = nn.ModuleList()
        self.input_size = input_size
        for k in kernelsizes:
            pad1 = int((k[0]-1)/2)
            pad2 = int((k[1]-1)/2)
            self.layers.append(nn.Sequential(OrderedDict([
                ('Aconv1', nn.Conv1d(
                    1, channels[0], k[0], padding=pad1, padding_mode='replicate')),
                ('Adrop1', nn.Dropout(dropout)),
                ('Atanh1', nn.Tanh()),
                ('Amaxp1', nn.MaxPool1d(kernel_size=2, stride=2)),

                ('Aconv2', nn.Conv1d(
                    channels[0], channels[1], k[1], padding_mode='replicate', padding=pad2)),
                ('Adrop2', nn.Dropout(dropout)),
                ('Atanh2', nn.Tanh()),
                ('Amaxp2', nn.MaxPool1d(kernel_size=2, stride=2)),

                ('Aflatten1', nn.Flatten()),

                ('Alin1', nn.Linear(int(channels[1] * input_size / 4), 32)),
                ('Aelu1', nn.ELU()),
                ('Alin2', nn.Linear(32, 1))
            ])))
        self.final_layer = nn.Sequential(OrderedDict([
            ('AlinF1', nn.Linear(len(kernelsizes), 10)),
            ('AeluF', nn.ELU()),
            ('AlinF2', nn.Linear(10, 1))
        ]))

    def forward(self, x):
        mid = torch.stack([l(x) for l in self.layers])
        out = self.final_layer(mid.view(mid.shape[1], -1))
        return out


class ConvCrusherNetPartB(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('Blin1', nn.Linear(input_size, input_size)),
            ('Belu1', nn.ELU()),
            ('Blin2', nn.Linear(input_size, 1)),
            ('Bsigm2', nn.Sigmoid())
        ]))

    def forward(self, x):
        x = self.layers(x)
        return x


class ConvCrusherNetPartC(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('Clin1', nn.Linear(input_size, input_size)),
            ('Celu1', nn.ELU()),
            ('Clin2', nn.Linear(input_size, 1)),
            ('Csigm2', nn.Sigmoid())
        ]))

    def forward(self, x):
        x = self.layers(x)
        return x


def getStrategy(price):
    if (price < -0.5):
        return 0
    if (price > 0.5):
        return 1
    if (price < 0):
        return 0.4
    return 0.6


def main():
    cont = False
    # init data
    if 1 > 0:
        take = [128, 64, 32, 16, 8]
        #take = [128, 64, 32]
        netA = {}
        ticker = ["MSFT", "AAPL", "T", "GOOGL", "WMT", "AMZN", "XOM", "CVX", "JPM", "GM",
                  "VZ", "PSX", "CMcSA", "WFC", "C", "TGT", "IBM", "DELL", "BA", "INTC", "FB", "FDX"]
        #ticker = ['MSFT', 'GOOGL']
        close = {}
        for t in ticker:
            if not (path.exists('data/close/'+t+'.csv')):
                c = yf.Ticker(t).history(start='2000-06-06',
                                         end='2019-06-06', interval="1d")['Close']
                c.to_csv('data/close/'+t+'.csv')

            if not (path.exists('data/perc/'+t+'.csv')):
                c = pd.read_csv('data/close/'+t+'.csv',  index_col=0)
                for i in range(0, len(c)-1):
                    c['Close'][len(c)-1-i] = 100 * (c['Close']
                                                    [len(c)-1-i] / c['Close'][len(c)-2-i] - 1)
                c[1:].to_csv('data/perc/'+t+'.csv')

            close[t] = pd.read_csv('data/perc/'+t+'.csv',  index_col=0)

        keys = close['MSFT'].index
        v = 0.2
        data_x, data_y, val_x, val_y = {}, {}, {}, {}
        if not path.exists('data/valindices.npy'):
            np.save('data/valindices.npy',
                    np.random.permutation(keys)[:int(v*len(keys))])
        val_indices = np.load('data/valindices.npy', allow_pickle=True)
    for t in take:
        # init and split data
        if 1 > 0:
            data_x[t], data_y[t], val_x[t], val_y[t] = {}, {}, {}, {}

            for tick in ticker:
                if not (path.exists('data/data'+str(t)+tick+'.npy')):
                    l = len(close[tick]['Close'])

                    data_x[t][tick] = [close[tick]['Close'][i: i+t].to_numpy()
                                       for i in range(np.max(take)-t, l-t) if close[tick].index[i+t] not in val_indices]
                    val_x[t][tick] = [close[tick]['Close'][i: i+t].to_numpy()
                                      for i in range(np.max(take)-t, l-t) if close[tick].index[i+t] in val_indices]

                    data_y[t][tick] = [[close[tick]['Close'][i+t]]
                                       for i in range(np.max(take)-t, l-t) if close[tick].index[i+t] not in val_indices]
                    val_y[t][tick] = [[close[tick]['Close'][i+t]]
                                      for i in range(np.max(take)-t, l-t) if close[tick].index[i+t] in val_indices]

                    np.save('data/data_x_'+str(t) +
                            tick+'.npy', data_x[t][tick])
                    np.save('data/val_x_'+str(t)+tick+'.npy', val_x[t][tick])
                    np.save('data/data_y_'+str(t) +
                            tick+'.npy', data_y[t][tick])
                    np.save('data/val_y_'+str(t)+tick+'.npy', val_y[t][tick])

                data_x[t][tick] = np.load('data/data_x_'+str(t)+tick +
                                          '.npy', allow_pickle=True)
                val_x[t][tick] = np.load(
                    'data/val_x_'+str(t)+tick+'.npy', allow_pickle=True)
                data_y[t][tick] = np.load('data/data_y_'+str(t)+tick +
                                          '.npy', allow_pickle=True)
                val_y[t][tick] = np.load(
                    'data/val_y_'+str(t)+tick+'.npy', allow_pickle=True)

            data_x[t] = torch.FloatTensor(np.concatenate(
                [data_x[t][tick] for tick in ticker])).float().reshape(-1, 1, t)
            data_y[t] = torch.FloatTensor(np.concatenate(
                [data_y[t][tick] for tick in ticker])).float().reshape(-1, 1)
            val_x[t] = torch.FloatTensor(np.concatenate(
                [val_x[t][tick] for tick in ticker])).float().reshape(-1, 1, t)
            val_y[t] = torch.FloatTensor(np.concatenate(
                [val_y[t][tick] for tick in ticker])).float().reshape(-1, 1)

        # init model A
        if 1 > 0:
            netA[t] = ConvCrusherNetPartA([32, 16], t, [[13, 7], [13, 5], [13, 3], [11, 7], [11, 5], [11, 3], [
                9, 7], [9, 5], [9, 3], [7, 7], [7, 5], [7, 3], [5, 7], [5, 5], [5, 3]], dropout=0.5).to(device)
            criterionA = nn.MSELoss()
            optimizerA = torch.optim.AdamW(netA[t].parameters(), lr=0.0001, betas=(
                0.9, 0.99), eps=0.0000000001, weight_decay=0.1)
            if (cont == True):
                p = 'model/modelA'+str(t)
                if (path.exists(p)):
                    netA[t].load_state_dict(p)

        # train model A
        if 1 > 0:
            batchsize = 256 * int(128 / t)
            epochs = 100
            amount = data_y[t].shape[0]
            amount_val = val_y[t].shape[0]
            cnt = int(amount / batchsize)+1
            cnt_val = int(amount_val / batchsize)+1
            fail = 0
            permu = np.random.permutation(range(amount))
            print('\r\n\r\n\r\n----------------------------------------------------------------------------------------')
            print('start training on previous {} days'.format(t))
            print('variance on train data is : {:3.2f} - on validation data is : {:3.2f}'.format(
                data_y[t].var(), val_y[t].var()))
            loss_train = []
            loss_val = []
            min_val_loss = 5000
            for e in range(epochs):
                if (fail <= 3):
                    if (e % 5 == 0):
                        netA[t].eval()

                        with torch.no_grad():
                            for j in range(cnt_val):
                                bxval, byval = val_x[t][batchsize*j:batchsize*(j+1)].to(
                                    device), val_y[t][batchsize*j:batchsize*(j+1)].to(device)
                                loss_val.append(criterionA(
                                    netA[t](bxval), byval))
                        print('Epoch : {:3} - train loss : {:3.2f} - val loss : {:3.2f}'.format(
                            e, torch.mean(torch.FloatTensor(loss_train)), torch.mean(torch.FloatTensor(loss_val))))
                        if torch.mean(torch.FloatTensor(loss_val)) < min_val_loss:
                            min_val_loss = torch.mean(
                                torch.FloatTensor(loss_val))
                            p = 'model/modelA'+str(t)
                            print('new best val loss, saving as {}'.format(p))
                            torch.save(netA[t].state_dict(), p)
                            fail = 0
                        else:
                            fail += 1
                        loss_train = []
                        loss_val = []

                    for i in range(cnt):
                        netA[t].train()
                        bx, by = data_x[t][permu[batchsize*i:batchsize*(i+1)]].to(
                            device), data_y[t][permu[batchsize*i:batchsize*(i+1)]].to(device)
                        optimizerA.zero_grad()
                        out = netA[t](bx)
                        loss = criterionA(out, by)
                        loss_train.append(loss)
                        loss.backward()
                        optimizerA.step()
            torch.save(netA[t].state_dict(), 'model/modelA'+str(t)+'final')
            netA[t].eval()

    # majority vote
    if 1 > 0:
        batchsize = 16
        amount_val = val_y[t].shape[0]
        cnt = int(amount_val / batchsize)  # left last uncomplete batch
        trades = strades = correctA = correctM = 0
        for i in range(cnt):
            with torch.no_grad():
                bx = torch.stack([netA[t](val_x[t][batchsize*i:batchsize*(i+1)].to(device))
                                  for t in take]).permute(1, 0, 2).reshape(batchsize, len(take))
                by = val_y[take[0]][batchsize*i:batchsize *
                                    (i+1)].to(device).reshape(batchsize)
                outA = torch.stack([sum(b) / len(take) for b in bx])
                outM = torch.stack([sum(b > 0) - sum(b < 0) for b in bx])
                trades += batchsize
                correctA += sum(torch.logical_and(outA >= 0, by >= 0)) + sum(
                    torch.logical_and(outA <= 0, by <= 0))
                correctM += sum(torch.logical_and(outM >= 0, by >= 0)) + \
                    sum(torch.logical_and(outM <= 0, by <= 0))
        print('Average vote correct: {} %'.format((100*correctA) / trades))
        print('Majority vote correct: {} %'.format((100*correctM) / trades))

    # init model B
    if 1 > 0:
        netB = ConvCrusherNetPartB(len(take)).to(device)
        criterionB = nn.MSELoss()
        optimizerB = torch.optim.AdamW(netB.parameters(), lr=0.001, betas=(
            0.9, 0.99), eps=0.0000000001)

    # train model B
    if 1 > 0:
        strong = 0.15
        batchsize = 256
        epochs = 20
        amount = data_y[t].shape[0]
        amount_val = val_y[t].shape[0]
        cnt = int(amount / batchsize)+1
        cnt_val = int(amount_val / batchsize)+1

        permu = np.random.permutation(range(amount))
        print('\r\n\r\n\r\n----------------------------------------------------------------------------------------')
        print('start training model B on all data')
        loss_train = []
        loss_val = []

        for e in range(epochs):
            if (e % 5 == 0):
                netB.eval()
                with torch.no_grad():
                    trades = strades = correct = scorrect = 0
                    for j in range(cnt_val):
                        bxval = torch.stack(
                            [netA[t](val_x[t][batchsize*j:batchsize*(j+1)].to(device)) for t in take]).permute(1, 0, 2).reshape(-1, len(take))
                        byval = val_y[t][batchsize*j:batchsize*(j+1)].detach().clone().apply_(
                            lambda y: getStrategy(y)).to(device)
                        # byval.apply_(lambda y: getStrategy(y)).to(device)
                        outB = netB(bxval)
                        loss_val.append(criterionB(outB, byval))
                        trades += batchsize
                        strades += sum(outB > 0.5 + strong) + \
                            sum(outB < 0.5 - strong)
                        correct += sum(torch.logical_and(outB >= 0.5, byval >= 0.5)) + \
                            sum(torch.logical_and(
                                outB <= 0.5, byval <= 0.5))
                        scorrect += sum(torch.logical_and(outB >= 0.5 + strong, byval >= 0.5)) + sum(
                            torch.logical_and(outB <= 0.5 - strong, byval <= 0.5))

                print('Epoch : {:3} - train B : {:3.2f}  - val B : {:3.2f} - correct% : {:3.2f} - strades : {} - scorrect% : {:3.2f}'.format(
                    e, torch.mean(torch.FloatTensor(loss_train)), torch.mean(torch.FloatTensor(loss_val)), (100 * correct / trades).item(), strades.item(), (100*scorrect / strades).item()))

                loss_train = []
                loss_val = []

            for i in range(cnt):
                netB.train()
                bx = torch.stack([netA[t](data_x[t][permu[batchsize*i:batchsize*(i+1)]].to(device))
                                  for t in take]).permute(1, 0, 2).reshape(-1, len(take))
                by = data_y[t][permu[batchsize*i:batchsize*(i+1)]
                               ].apply_(lambda y: getStrategy(y)).to(device)
                optimizerB.zero_grad()
                outB = netB(bx)
                lossB = criterionB(outB, by)
                loss_train.append(lossB)
                lossB.backward()
                optimizerB.step()

        torch.save(netB.state_dict(), 'model/modelB')
        netB.eval()

    # init model C
    if 1 > 0:
        netC = ConvCrusherNetPartC(len(take)).to(device)
        criterionC = nn.MSELoss()
        optimizerC = torch.optim.AdamW(netC.parameters(), lr=0.001, betas=(
            0.9, 0.99), eps=0.0000000001)
    # train model C

    if 1 > 0:
        batchsize = 256
        epochs = 100
        amount = data_y[t].shape[0]
        amount_val = val_y[t].shape[0]
        cnt = int(amount / batchsize)+1
        cnt_val = int(amount_val / batchsize)+1

        permu = np.random.permutation(range(amount))
        print('\r\n\r\n\r\n----------------------------------------------------------------------------------------')
        print('start training on all data')
        loss_train = []
        loss_val = []

        for e in range(epochs):
            if (e % 5 == 0):
                netC.eval()
                with torch.no_grad():
                    trades = strades = correct = scorrect = 0
                    for j in range(cnt_val):
                        cxval = torch.stack(
                            [netA[t](val_x[t][batchsize*j:batchsize*(j+1)].to(device)) for t in take]).permute(1, 0, 2).reshape(-1, len(take))
                        cyval = val_y[t][batchsize*j:batchsize *
                                         (j+1)].sigmoid().to(device)
                        outC = netC(cxval)
                        loss_val.append(criterionC(outC, cyval))
                        trades += batchsize
                        strades += sum(outC > 0.5 + strong) + \
                            sum(outC < 0.5 - strong)
                        correct += sum(torch.logical_and(outC >= 0.5, cyval >= 0.5)) + \
                            sum(torch.logical_and(
                                outC <= 0.5, cyval <= 0.5))
                        scorrect += sum(torch.logical_and(outC >= 0.5 + strong, cyval >= 0.5)) + sum(
                            torch.logical_and(outC <= 0.5 - strong, cyval <= 0.5))

                print('Epoch : {:3} - train C : {:3.2f}  - val C : {:3.2f} - correct% : {:3.2f} - strades : {} - scorrect% : {:3.2f}'.format(
                    e, torch.mean(torch.FloatTensor(loss_train)), torch.mean(torch.FloatTensor(loss_val)), (100 * correct / trades).item(), strades.item(), (100*scorrect / strades).item()))

                loss_train = []
                loss_val = []

            for i in range(cnt):
                netC.train()
                cx = torch.stack([netA[t](data_x[t][permu[batchsize*i:batchsize*(i+1)]].to(device))
                                  for t in take]).permute(1, 0, 2).reshape(-1, len(take))
                cy = data_y[t][permu[batchsize*i:batchsize *
                                     (i+1)]].sigmoid().to(device)
                optimizerC.zero_grad()
                outC = netC(cx)
                lossC = criterionC(outC, cy)
                loss_train.append(lossC)
                lossC.backward()
                optimizerC.step()

        torch.save(netC.state_dict(), 'model/modelC')
        netC.eval()


if __name__ == '__main__':
    main()
