from numpy import float32, zeros_like
import torch
import pandas as pd
from datetime import datetime
from data.starting_kit.ts_split import GroupedTimeSeriesSplit
import matplotlib.pyplot as plt


def filt(wave, beta1=0.01, beta2=0.001, n:int=1):
        waves = []
        for dwelling in range(0, 61):
            #fft = torch.fft.fft(wave[dwelling])
            #plt.plot(fft, label="fft")
            #plt.show()
            w = wave[dwelling].clone()

            #filtering
            # for f in [5, 6, 904, 905]:
            #     f_wave = filt(w, 1, f)
            #     #plt.plot(f_wave, label=f"filtered with {f}")
            #     indices = (f_wave < 1).nonzero() + 1

            #     w[indices] = w[indices + 1]

            #plt.plot(wave[dwelling] / torch.max(wave[dwelling]), label="unfiltered")
            dt = wave[dwelling, :-1] - wave[dwelling, 1:]
            for i in range(2,n):
                dt += torch.cat([wave[dwelling, :-i] - wave[dwelling, i:], torch.tensor([0]*(i-1))])
            dt /= n+1
            n_dt = dt / torch.max(dt)

            #plt.plot(n_dt)
            #plt.plot(n_dt ** 2)
            

            n_ = (n_dt ** 2 * torch.sign(n_dt) > beta1)
            n_ = torch.concat([n_, torch.tensor([0])])

            indices = (n_ > beta2).nonzero() + 1

           
            indices[indices>=(wave.shape[-1] - n)] -= n+1
            indices[indices<=(n-1)] += n
             # linear fill
            forward = w[indices + n]
            backward = w[indices - n]
            w[indices] = (forward + backward) / 2
            waves.append(w)
        return torch.stack(waves, dim=0)

def clean_data():
    df = pd.read_csv('data/starting_kit/train.csv')
    # drop index for feature preparation
    df = df.drop(columns='pseudo_id')
    # convert dates to pandas datetime
    df.columns = [datetime.strptime(c, "%Y-%m-%d %H:%M:%S") for c in df.columns]
    # Aggregate energy use values per day
    df = df.T
    df.index = pd.to_datetime(df.index)

    tscv = GroupedTimeSeriesSplit(train_window= 912 * 2, test_window=168 * 2, train_gap = 0, freq="30min")

    for train_ind, test_ind in tscv.split(df, y=df, dates = df.index):
        wave = torch.tensor(df.iloc[train_ind].values, dtype=torch.float32).transpose(0, 1)
        og = wave.clone().detach()
        wave = filt(wave, 0.001, 0.0001, 1)

        plt.plot(og[0], label="original")
        plt.plot(wave[0], label="filtered")

        wave = filt(wave, 0.0001, 0.00001)

        plt.plot(wave[0], label="double_filtered")

        wave = filt(wave, 0.00001, 0.000001)

        plt.plot(wave[0], label="tripple_filtered")
        plt.legend(loc="upper right")
        plt.show()
        for dwelling in range(0, 61):
            f_ = torch.fft.fft(wave[dwelling])
            l = 100
            fill = f_[456-l:456+l].clone()
            f_[50:-50] *= 0
            f_[456-l:456+l] = fill
            plt.plot(f_)
            plt.show()
            # f_[:2] *= 0
            f_filt = torch.fft.ifft(f_)
            plt.plot(f_filt, label="filtered")
            plt.plot(wave[dwelling], label="unfiltered")
            #plt.plot((wave[dwelling] + f_filt) / 2)
            plt.show()


        #plt.plot((n_ > 0.05) * 100.0)
        #plt.plot(n_ > .05, label="gradient magnitude")


# Ignore.
# this uses torch audio and is not required for the filtering that we used for the final solution.        

# def clean_with_fft(in_f):
#     df = pd.read_csv(in_f)

#     tscv = GroupedTimeSeriesSplit(train_window= 912, test_window=168, train_gap = 0, freq="H")
#     filt = ta.functional.highpass_biquad
    
#     for train_ind, test_ind in tscv.split(df, y=df, dates = df.index):
#         wave = torch.tensor(df.iloc[train_ind].values, dtype=torch.float32).transpose(0, 1)
#         for dwelling in range(0, 61):
#             f_ = torch.fft.fft(wave[dwelling])
#             plt.plot(f_)
#             plt.show()
#             f_[50:-50] *= 0
#             f_[5] *= 0
#             f_filt = torch.fft.ifft(f_)
#             plt.plot(f_filt, label="filtered")
#             plt.plot(wave[dwelling], label="unfiltered")
#             #plt.plot((wave[dwelling] + f_filt) / 2)
#             plt.show()


def clean_data_lerp(in_f, out_f):
    df = pd.read_csv(in_f)
    # drop index for feature preparation
    df = df.drop(columns='pseudo_id')
    # convert dates to pandas datetime
    df.columns = [datetime.strptime(c, "%Y-%m-%d %H:%M:%S") for c in df.columns]
    # Aggregate energy use values per day
    df = df.T
    df.index = pd.to_datetime(df.index)

    tscv = GroupedTimeSeriesSplit(train_window= 912 * 2, test_window=168 * 2, train_gap = 0, freq="30min")

    def lerp_zeros(wave, keepmodes=15, tolerance = 0.15, eps=0.01, methode="fft", plot=False): #, tolerance=0.25, dropmodes=40):
        pl_sv = plot
        waves = []
        for dwelling in range(0, 61):
            plot = pl_sv
            w = wave[dwelling].clone()
            scale = torch.max(w)
            w /= scale

            if methode=="eps":      
                zeros = (w<=eps).nonzero()

            else:
                fft = torch.fft.fft(w)
                fft[:1] *= tolerance
                if keepmodes:
                    fft[keepmodes:-keepmodes] *= 0
                thresh = torch.fft.ifft(fft)
                thresh *= 0.7 # "squash" a little in y direction to avoid to sharp peaks
                zeros = (w<=thresh.real).nonzero()
            
            if zeros.shape[0] > 0.3*w.shape[0]:
                print("warning... tried to cut of more than 30% of datapoints")
                print(f"dwelling: {dwelling}")
                plot = True

            if plot:
                plt.plot(w * scale, label="original", c=[0.0, 0.2, 0.8, 0.3])
                plt.plot(thresh * scale, label="decision boundary", c=[1.0, 0.0, 0.0, 0.6])
                plt.scatter(zeros, w[zeros] * scale, c="r")
                plt.show()
            
            if zeros.squeeze(-1).shape[0] < 1:
                f"nothing was removed in dwelling: {dwelling} using method: {methode}"
                waves.append(w*scale)
                continue

            last = zeros.squeeze(-1)[0]
            seq_l = 0

            if len(zeros) > 1:
                for i in zeros.squeeze(-1):
                    if i != last + 1:
                        i_s = max(last-seq_l, 0)
                        if seq_l <= 1:
                            w[last] = (w[last-1] + w[last+1]) / 2
                        else:
                            #print(f"lerp between [{i_s}]:{w[i_s]} -> [{last+1}]: {w[last+1]} ")
                            if i_s == 0:
                                # consecutive zeros at the start are suspicious... we should just average them out
                                print(f"skip removal of consecutive beginning at: dwelling {dwelling}")
                                plot = True
                                #w[i_s:last+1] = torch.mean(w[i_s], w[last+1])
                            else:
                                li = torch.linspace(w[i_s], w[last+1], seq_l+1)
                                #print(li)
                                w[i_s:last+1] = li
                        seq_l = 1
                    else:
                        seq_l += 1
                    last = i
            
            if last < (w.shape[0] - 1):
                i_s = last-seq_l
                if seq_l == 1:
                    w[last] = (w[last-1] + w[last+1]) / 2
                else:
                    #print(f"lerp between [{i_s}]:{w[i_s]} -> [{last+1}]: {w[last+1]} ")
                    li = torch.linspace(w[i_s], w[last+1], seq_l+1)
                    #print(li)
                    w[i_s:last+1] = li
                seq_l = 1
            
            if plot:
                plt.plot(w * scale, label="zeros removed", c=[1.0, 0.5, 0.1, 1.0])
                plt.legend(loc="upper right")
                plt.show()
        
            waves.append(w*scale)
        return torch.stack(waves, dim=0)

    for train_ind, test_ind in tscv.split(df, y=df, dates = df.index):
        wave = torch.tensor(df.iloc[train_ind].values, dtype=torch.float32).transpose(0, 1)
        #uf = wave.clone()
        
        wave = lerp_zeros(wave, methode="eps", plot=False)
        #lpz = wave.clone()

        wave = filt(wave, beta1=0.3, beta2=0.1)
        #grd = wave.clone()

        wave = lerp_zeros(wave, methode="eps", plot=False)

        df.iloc[train_ind] = wave.transpose(0, 1).numpy()
    
    df = df.T
    df.index.name = 'pseudo_id'
    print(df)
    print(f"writing filtered output to: {out_f}")
    df.to_csv(out_f)


if __name__ == "__main__":
    #clean_with_fft()
    #clean_data()
    clean_data_lerp()