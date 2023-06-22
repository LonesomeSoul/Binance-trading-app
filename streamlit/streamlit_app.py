#!/usr/bin/env python
# coding: utf-8

# In[1]:


from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os,re
import trading_data,trading_lstm, trading_bot

import tkinter as tk
from tkinter import filedialog

from binance.spot import Spot 
from sys import stderr
import sys, os,datetime,requests,json,pandas as pd,numpy as np
import time,math, gc

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn.model_selection import train_test_split
min_max_scaler = sklearn.preprocessing.MinMaxScaler()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



# In[2]:


#directory=r"G:\Учеба\биржа\streamlit_data"
directory="streamlit_data"
selected_folder=directory
currency_name="ETHUSDT"

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    #st.title("AutoNickML")
    choice = st.radio("Navigation", ["Choose currency","Data description","Bot training","Bot backtesting","Bot using"])
    #st.info("This project application helps you build and explore your data.")
    
if choice=="Choose currency":
    st.title("Choosing and downloading currency data")
    currency_name=st.text_input(label="Input chosen currency pair here. Example: ETH/USDT")
    
    if (currency_name):
        currency_name=re.sub('[^0-9a-zA-Z]+', '', currency_name)
        days_downloading=st.text_input(label="Set how many days you want the bot to train on")
        if (days_downloading):
            try:
                days_downloading=int(days_downloading)
                os.makedirs(os.path.join(directory,currency_name),exist_ok=True) 
                downloaded_data=trading_data.download_klines(currency_name,os.path.join(directory,currency_name),days_downloading,target_count=4)
                
                if (os.path.exists(os.path.join(directory,currency_name,f"{currency_name}.csv"))):
                    count=0
                    for file_name_item in os.listdir(directory):
                        if os.path.splitext(file_name_item)[1]==".csv":
                            count+=1
                    downloaded_data.to_csv(os.path.join(directory,currency_name,f'{currency_name}_копия_{count}.csv'),index=False)
                else:
                    downloaded_data.to_csv(os.path.join(directory,currency_name,f'{currency_name}.csv'),index=False)
                st.text("Your data has been downloaded successfully!")
                st.dataframe(downloaded_data.head(5))
                displayed_chart=downloaded_data["Open"][-3000:].plot().get_figure()
                displayed_chart.savefig(os.path.join(directory,currency_name,f"{currency_name}_chart.jpg"))
                st.image(os.path.join(directory,currency_name,f"{currency_name}_chart.jpg"))
            except:
                print("error")
                raise

if choice=="Bot training":
    scaling=1
    st.title("Uploading the data to train the bot on")
    file_train=st.file_uploader(label="Upload the data")
    if (file_train):
        
        file_train_name=os.path.splitext(file_train.name)[0].split(r"/")[-1]
        os.makedirs(os.path.join(directory,file_train_name),exist_ok=True) 
        X_train, X_test, y_train, y_test, X_train_opens, X_test_opens=trading_lstm.prepare_data_for_train(file_train)
        net,net_min_err_train,net_min_err_test,net_max_std,err_df,stds_single=trading_lstm.train_bot( X_train, X_test, y_train, y_test,X_train_opens, X_test_opens,epochs=100)
        model_scripted = torch.jit.script(net_min_err_test) # Export to TorchScript
        model_scripted.save(os.path.join(directory,file_train_name,f'{file_train_name}.pt')) # Save
        displayed_chart_train=err_df.plot().get_figure()
        displayed_chart_train.savefig(os.path.join(directory,file_train_name,"bot_losses_chart.jpg"))
        st.image(os.path.join(directory,file_train_name,"bot_losses_chart.jpg"))
        
if choice=="Bot backtesting":
    model_backtest=st.file_uploader(label="Upload the model for backtest")
    data_backtest=st.file_uploader(label="Upload the data for backtest")
    if (model_backtest and data_backtest):
        net=torch.jit.load(model_backtest,map_location="cpu")
        net.eval()
        X_train, X_test, y_train, y_test, X_train_opens, X_test_opens=trading_lstm.prepare_data_for_train(data_backtest,device="cpu")
        backtest_results=[]
        backtest_results_means=[]
        stop_loss=0.15
        for buy in np.arange(0.001,0.01,0.001):
            print(buy)
            for sell in np.arange(0.001,0.01,0.001):
                for net_repeats in range(1):
                    for stop in np.arange(0.8,0.95,0.02):
                        arr=[]
                        for i in range(1):
                            backtest_results.append([trading_lstm.backtest(net,X_train,X_train_opens,min_sell_threshold=1+sell,buying_threshold=1+buy,stop_loss=stop),stop_loss])
        array=[]
        for i in backtest_results:
            if len(i[0])>0:
                array.append(np.array(list(map(lambda x: x.item() , i[0]))))
        array=np.array(array)
        res_array=[]
        for i in array:
            if (len(i)>0):
                res_array.append(i[-1])
        res_array=np.array(res_array)
        max_res_pos=np.argmax(res_array)
        
        backtest_plot=pd.DataFrame(array[max_res_pos]).plot().get_figure()
        backtest_plot.savefig(os.path.join(directory,os.path.dirname(model_backtest.name),"bot_backtest.jpg"))
        st.image(os.path.join(directory,os.path.dirname(model_backtest.name),"bot_backtest.jpg"))
    
if choice=="Bot using":
    model_bot_using=st.file_uploader(label="Upload the bot model")
    currency_name_bot=st.text_input(label="Input chosen currency pair or leave empty. Example: ETH/USDT")
    if (currency_name_bot):
        currency_name_bot=re.sub('[^0-9a-zA-Z]+', '', currency_name_bot)
    if (model and currency_name_bot==""):
        currency_name_run=os.path.splitext(model_bot_using.name)[0].split(r"/")[-1]
        os.makedirs(os.path.join(directory,currency_name_run),exist_ok=True) 
        trading_bot.run_bot(symbol=currency_name_run,directory="bot_data", net_file=model_bot_using)
    if (model and currency_name_bot):
        os.makedirs(os.path.join(directory,currency_name_bot),exist_ok=True) 
        trading_bot.run_bot(symbol=currency_name_bot,directory="bot_data", net_file=model_bot_using)
        

if choice=="Data description":
    file_data_description=st.file_uploader(label="Upload the data")
    if (file_data_description):
        df=pd.read_csv(file_data_description).reset_index(drop=True)
        st_profile_report(df.profile_report())


# In[ ]:




