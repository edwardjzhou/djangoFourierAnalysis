from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

import statsmodels.api as sm 
from statsmodels import regression 

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

import scipy as sp
import scipy.fftpack
import json
class Portfolio:
    def __init__(self, xydict, marketindex = 'SPY'):
        self.portfolio = dict()
        self.marketindex = marketindex #can use 'VOO' but only back to 9/2010; 'SPY' goes back to 2/1993
        self.closespctchange = []
        self.betas = []
        self.alphas = []
        for key in xydict:
            self.portfolio[key] = xydict[key]
        print(self.portfolio)

    def add(self, xydict):
        for ele in xydict:
            self.portfolio[ele] = xydict[ele]
            print("Added", ele,"since", xydict[ele])
        print(self.portfolio)
    
    def delete(self, deletekey):
        try:
            del self.portfolio[deletekey]
            print("Deleted " + deletekey)
        except (KeyError): # THIS HAS TO BE A TUPLE IF YOU WANT MORE THAN 1 ERROR
            print("Key 'deletekey' not found")
        finally:
            print(self.portfolio)

    
    def beta(self):
        today = date.today().strftime("%Y-%m-%d")     
        for key in self.portfolio:
            if len(self.portfolio[key]) == 2:
                marketdf = pdr.get_data_yahoo(self.marketindex, start = self.portfolio[key][0], 
                                              end = self.portfolio[key][1] )
                stockdf = pdr.get_data_yahoo(key, start = self.portfolio[key][0], end = self.portfolio[key][1])
            else:
                marketdf = pdr.get_data_yahoo(self.marketindex, start = self.portfolio[key], end = today)
                stockdf = pdr.get_data_yahoo(key, start = self.portfolio[key], end = today)
            return_market = marketdf.Close.pct_change()[1:] #~253 trading days per year - 1, the first day will be NaN
            return_stock = stockdf.Close.pct_change()[1:]
            self.closespctchange.append(return_stock) #for self.fourier() to use later
            # plt.figure(figsize=(20,10)) 
            # return_market.plot(color="lavender")
            # return_stock.plot(color="blue") 
            # plt.ylabel("Daily Returns (percentage pts) of "+ key +" and " + self.marketindex) 
            # or *100 to get % change of Day-on-Day Closes
            # L=plt.legend()
            # L.get_texts()[0].set_text(self.marketindex)
            # L.get_texts()[1].set_text(key)
            # plt.show()
            X=return_market.values 
            Y=return_stock.values 
            alpha, beta = self.linreg(X,Y) 
            # if len(self.portfolio[key]) == 2:
            #     print(key + " from " + self.portfolio[key][0] + " to " + self.portfolio[key][1])
            # else:
            #     print(key + " from " + self.portfolio[key] + " to today.")
            # print('alpha: ' + str(alpha))
            # print('beta: ' + str(beta)) 
            self.alphas.append(alpha) # stored as part of the object
            self.betas.append(beta)
            return self.alphas, self.betas
            
    def linreg(self,x,y): 
        x = sm.add_constant(x) 
        model = regression.linear_model.OLS(y,x).fit()
        x = x[:,1]  #tuple slice takes 2nd ele of every vector of vectors
        return model.params[0],model.params[1]
    
    def fourier(self):
        for index, ele in enumerate(self.closespctchange,start=0):
            basecase=[]
            for ele1 in ele:
                basecase.append(ele1)
            ele_fft = sp.fftpack.fft(ele)
            ele_psd = np.abs(ele_fft) ** 2
            fftfreq = sp.fftpack.fftfreq(len(ele_psd), 1. / len(ele_psd)) 
            # fftfreq(n samples,1/n spacing)
            # entire waveform must be visible; 
            # n samples ( n trading days) ~253 trading days/year
            # highest frequency we can see (full sin happens once / 2 trading days time) and is liable to be fully noise
            # which caps highest frequency at (number of trading days / 2) (times / entire time)
            # lowest frequency we can see is (something that happens once full sin / entire time), or (1 time / n trading days)
            # frequency of 0 is technically the average value entire time, but we don't see
            i = fftfreq > 0 #T/F per index
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.plot(fftfreq[i], ele_psd[i], 'ro') #plot positive frequencies
            
            global power 
            power = ele_psd[i].copy()
            global frequencies 
            frequencies = fftfreq[i].copy()
            
            ax.set_xlim(-5, (len(ele_psd)/2)+10)
            ax.set_xlabel('Frequency (times/total sampling range of '+str(len(ele_psd))+' or ' + str(~int(len(ele_psd)/253)) + ' years of trading days)')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title(list(self.portfolio)[index] + " training data")
            # end first plot above, start second plot below
            
            
            ele_fft_bis = ele_fft.copy()
            
            global GLOBAL_FFT_BIS
            GLOBAL_FFT_BIS = ele_fft_bis.copy()
            global GLOBAL_FFTFREQ
            # GLOBAL_FFTFREQ = fftfreq.copy()
            # plt.figure(figsize=(20,8)) 

            for i in range(106,120):
                ele_fft_bis= ele_fft.copy()
                ele_fft_bis[np.abs(fftfreq) != 107  ] = 0 # len(ele_psd)/2 is perfect matching with (trading days/2) waves
                ele_slow = np.real(sp.fftpack.ifft(ele_fft_bis)) #inverse fourier transform of the frequencies
                plt.plot(ele_slow,'r--')
            #plt.plot(basecase,ele_slow,'b--') if perfect with n/2 waves, it should be a straight line
            # plt.plot(basecase,'b--')
            # #plt.plot(ele_slow,'ro')
            # plt.xlim(-1, len(ele_psd))
            # plt.ylabel('Return (percentage points)')
            # plt.xlabel('Trading days')
            # plt.title(list(self.portfolio)[index] + " training data")
            # plt.show()
            self.basecase = basecase
            self.eleslow = ele_slow


def index(request):
    pets = Portfolio({"PETS":"2018-10-10"})
    # petzx = Portfolio({"PETZX":"2018-10-10"})
    output = pets.beta()
    alpha =  str(output[0])
    beta = str(output[1])
    pets.fourier()
    basecase = str(pets.basecase)
    eleslow = str(pets.eleslow.tolist())
    # listToStr = ' '.join([str(elem) for elem in s]) 
    data = {
        "PETS": {
            "alpha": alpha,
            "beta": beta,
            "basecase": basecase,
            "eleslow": eleslow,
        }
    }
    return HttpResponse(json.dumps(data))

# Merck & Co. (MRK)
# CVS Health (CVS)
# ProShares Pet Care ETF (PAWZ)
# Tractor Supply (TSCO)
# Gabelli Pet Parents' Fund (PETZX)
# PetMed Express (PETS)
# Trupanion (TRUP)