#!/usr/bin/env python
# coding: utf-8

# In[28]:


import binance 
from sys import stderr
from binance import client
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
import bot_cfg


# In[7]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def import_api(time_last,symbol="BTCUSDT",limit=12*60,interval="1m"):
    params={"symbol":symbol,"limit":limit,"interval":interval,"endTime":time_last,"startTime":(time_last-12*3600*1000)}
    r=requests.get(url="https://api.binance.com/api/v3/klines", params=params)
    df=pd.DataFrame(r.json())
    return df

def transform_df (df):
    column_names=['Open_time','Open','High','Low','Close','Volume','Close_time','Quote_asset_volume',
               'Number_of_trades','Taker_buy_base_asset_volume','Taker_buy_quote_asset_volume','Ignore']
    df=df.reset_index(drop=True)
    df.set_axis(column_names,axis=1,inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(float)
    weekday=[]
    month=[]
    for i in range(df['Open'].size):
        dt_open=datetime.datetime.fromtimestamp(df['Open_time'][i]//1000)
        dt_close=datetime.datetime.fromtimestamp(df['Close_time'][i]//1000)
        df['Open_time'][i]=dt_open.hour*3600+dt_open.minute*60+dt_open.second
        df['Close_time'][i]=dt_close.hour*3600+dt_close.minute*60+dt_close.second
        weekday.append(dt_open.weekday())
        month.append(dt_open.month)
        #df['Open_time'][i]=dt_open.strftime("%I:%M:%S")
        #df['Close_time'][i]=dt_close.strftime("%I:%M:%S")
    df['weekday']=weekday
    df['month']=month
    df=df.drop(df.columns[[7,10,11,13]],axis=1)
    del(month)
    del(weekday)
    return df


def moving_average(data,range_,concat=60):
    leftover=len(data)%concat
    data=data[::concat].reset_index(drop=True)
    mean=[]
    arr=[]
    for i in range(range_):
        this_mean=0
        for j in range(i+1):
            this_mean+=data[j]
        mean.append(this_mean/(j+1))
    if leftover:
        data_size=len(data)-1
    else:
        data_size=len(data)
    for i in range(range_,data_size):
        this_mean=data[i]
        for j in range(1,range_):
            this_mean+=data[i-j]
        mean.append(this_mean/(range_))
    arr=leftover*[mean[0]]
    for i in range(len(mean)):
        arr+=concat*[mean[i]]
    
    
    return arr

def generate_previous(df,count,col_name):
    #idxes=
    df=df.reset_index(drop=True)
    values=list(df[col_name][0:count])
    columns={}
    for i in range(count):
        this_name="prev_"+col_name+"_"+str(i+1)
        columns[this_name]=[]
    for i in range(count,df[col_name].size):
        for j in range(count):
            this_name="prev_"+col_name+"_"+str(j+1)
            columns[this_name].append(values[-(j+1)])
        values.pop(0)
        values.append(df[col_name][i])
    size=df[col_name].size
    df=df[count:size]
    for i in range(count):
        this_name="prev_"+col_name+"_"+str(i+1)
        df[this_name]=columns[this_name]
    
    return df.reset_index(drop=True)

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [a for a in (df.iloc[i:i+window_size].drop("target",axis=1).values)]
        X.append(row)
        label = df["target"][i+window_size-1]
        y.append(label)
    return np.array(X), np.array(y)

def tensor_to_X_y(data, window_size=5):
    X = []
    y = []
    data=pd.DataFrame(data.tolist())
    columns=data.columns
    for i in range(len(data)-window_size):
        row = [a for a in (data.iloc[i:i+window_size].drop(data.columns[-1],axis=1).values)]
        X.append(row)
        label = data[data.columns[-1]][i+window_size-1]
        y.append(label)
    return np.array(X), np.array(y)

def preprocess_df(df,target_range,hours,hours_interval,days,days_interval,scaling_range=0.2):
    target=[]
    concat_hours=3
    concat_days=8
    window=target_range
    for i in range(df["Open"].size-window):
        target.append(df["Open"][i+window])
    df=df[0:df["Open"].size-window]
    df["target"]=target
    df["Open"]=df["Close"]
    df=df.drop(["Close_time","Taker_buy_base_asset_volume","Volume","Close","Low","High","Number_of_trades"],axis=1)
    for i in range(1,hours//hours_interval+1):
        df["mean_"+str(i*hours_interval)+"_hours"]=moving_average(df["Open"],i*(12//concat_hours)*hours_interval,concat_hours)
    for i in range(1,days//days_interval+1):
        df["mean_"+str(i*days_interval)+"_days"]=moving_average(df["Open"],i*12*(24//concat_days)*days_interval,concat_days)
    drop_col=["Open",'Open_time','weekday']
    open_delta=[]
    df["weekday"]=df["weekday"].astype(float)
    for col in df.drop(drop_col,axis=1).columns:
        for i in range(df["Open"].size):
            df[col][i]=df[col][i]/df["Open"][i]
    for i in range(1,df["Open"].size):
        df["Open_time"][i]/=86400
        df["weekday"][i]/=6.0
        delta=df["Open"][i]-df["Open"][i-1]
        open_delta.append(delta/df["Open"][i])
    df=df.drop(0,axis=0)
    cols=df.columns[3:]
    for i in cols:
        for j in range(1,df["Open"].size):
            df[i][j]=(df[i][j]-(1-scaling_range))/(scaling_range*2)
    return df

def upscale(input_data,scaling_range=0.2):
    return input_data*2*scaling_range-scaling_range+1


# In[8]:


class autoencoder(nn.Module):
    def __init__(self,drop,input_size,hidden_size,test_size):
        super(autoencoder, self).__init__()
        self.norm=nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc3 = nn.Linear(hidden_size, test_size,bias=True)
        self.fc4 = nn.Linear(test_size, hidden_size,bias=True)
        self.fc5 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc6 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc7 = nn.Linear(hidden_size, input_size,bias=True)
        self.dropout = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(F.relu(x))
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = self.dropout(F.relu(x))
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        return x
    def encode (self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x
    
class Net(nn.Module):
    def __init__(self,drop,hidden_size,input_size):
        super(Net, self).__init__()
        self.norm=nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.LSTM(num_layers=1,input_size=input_size, hidden_size=hidden_size,batch_first=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc4 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc5 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc6 = nn.Linear(hidden_size, 1,bias=True)
        self.dropout = nn.Dropout(drop)
    def forward(self, x):
        x,_ = self.fc1(x)
        x=x[:,-1,:]
        x=F.tanh(x)
        x = self.dropout(x)
        x = F.logsigmoid(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x
    def predict(self, x):
        x,_ = self.fc1(x)
        x=x[:,-1,:]
        x=F.tanh(x)
        x = self.dropout(x)
        x = F.logsigmoid(self.fc5(x))
        #x = self.dropout(x)
        x = self.fc6(x)
        return x


# In[9]:


def encode(encoder,data):
    encoded=encoder.encode(torch.tensor(data,device="cuda"))
    #if (encoded.shape.shape!=1):
    arr=torch.tensor([0 for i in encoded[:,0]],device="cuda")
    #print(encoded[:,1])
    count=0
    for i in range(encoded.shape[1]):
        if (set(encoded[:,i].tolist())!={0.0}):
            arr=torch.cat((arr,encoded[:,i]),0)
            count+=1
    arr=torch.reshape(arr,[count+1,encoded.shape[0]])[1:]
    arr=torch.reshape(arr,[encoded.shape[0],count])
    return arr

def tensor_to_sequences(data, window_size=5):
    X = []
    data=pd.DataFrame(data.tolist())
    for i in range(len(data)-window_size):
        row = [a for a in (data.iloc[i:i+window_size].values)]
        X.append(row)
    return np.array(X)


# In[10]:





# In[13]:


#count=2*365
#time_now=datetime.datetime.now()
#time_now=round(time_now.timestamp())*1000
#time_now-=24*3600*1000
#df1=pd.DataFrame()
#for i in range(2*31): #2*365
#    df1=import_api(time_now-12*3600*1000*i,symbol,720,"5m").append(df1)
#df1=transform_df(df1)
#df1=preprocess_df(df1,4,24,1,15,1).reset_index(drop=True)


# In[ ]:


def run_bot(net_file,symbol="ETHUSDT",directory="bot_data"):
    
    
    
    key_api=bot_cfg.key_api
    secret_api=bot_cfg.secret_api
    net=torch.jit.load(net_file)
    net.eval()
    
    count=2*365
    time_now=datetime.datetime.now()
    time_now=round(time_now.timestamp())*1000
    #time_now-=24*3600*1000
    df1=pd.DataFrame()
    for i in range(2*31): #2*365
        df1=import_api(time_now-12*3600*1000*i,symbol,720,"5m").append(df1)
    df1=transform_df(df1)
    df1=preprocess_df(df1,4,24,1,15,1).reset_index(drop=True)
    #encoder=autoencoder(0,41,64,10).cuda().double()
    #encoder.load_state_dict(torch.load(directory+'\\encoder_05-07-2022_22-03-48_64_10.pth'))
    window_size=30
    leftover=0
    beginning=-window_size-1
    df1_part=df1.iloc[beginning:]
    opens=df1_part["Open"]
    test_opens=np.asarray(df1_part["Open"][:])
    #encoded=encode(encoder,df1_part.drop(["Open","target"],axis=1).to_numpy()[:]).tolist()
    encoded=df1_part.drop(["Open","target"],axis=1).to_numpy()
    encoded=torch.tensor(encoded,device="cuda")
    seq=tensor_to_sequences(encoded[-(window_size+1):],window_size)
    test_data=tensor_to_sequences(encoded[:],window_size)
    for i in range(len(test_data)):
            for j in range(test_data[i].shape[0]):
                test_data[i][j]=np.array(test_data[i][j])
    
    client_api = client.Client(api_key=key_api, api_secret=secret_api)
    url = 'https://api.binance.com/'
    with requests.Session() as session:
        session.headers = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        session.headers = "Mozilla/5.0 (X11; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0"
        session.get(url)
    url = 'https://binance.com/'
    with requests.Session() as session:
        session.headers = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        session.headers = "Mozilla/5.0 (X11; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0"
        session.get(url)

    wallet_start=1000
    min_sell_threshold=1.003 #min value for selling
    buying_threshold=1.002 #min prediction for buying
    purchase_size=0.2
    purchase_size_usdt=10.5
    net_repeating=3
    comission=0.001
    stop_loss=0.85
    loss_multiplier=4
    loss_count_max=2

    wallet=wallet_start
    desired_sell_cost=0
    min_sell_value=0
    wallet_dynamics=[]
    buy=0
    backtest_preds=[]
    positions=[]
    log_time_count=0
    loss_count=0
    last_trade_data=pd.DataFrame({'price':[0], 'quantity':[0],"take_profit_price":[0],"stop_loss_price":[0]})
    while(True):
        try:
            if (log_time_count%4)==0:
                #count=0
                url = 'https://api.binance.com/'
                with requests.Session() as session:
                    session.headers = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                    session.headers = "Mozilla/5.0 (X11; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0"
                    session.get(url)
                url = 'https://binance.com/'
                with requests.Session() as session:
                    session.headers = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                    session.headers = "Mozilla/5.0 (X11; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0"
                    session.get(url)
            last_trade_data=pd.read_csv(directory+'\\last_trade_data.csv').astype(float)
            #buy=0
            if last_trade_data['price'][0]==0:
                buy=0
            else:
                buy=1

            side=""
            #orders=client.get_open_orders("ETHUSDT")
            time.sleep(0.3)
            #balances=client.account_snapshot ("SPOT")["snapshotVos"][-1]["data"]["balances"]
            #if (buy==0 and len(orders)==1):
            #    client.cancel_open_orders(symbol="ETHUSDT")
            time_now=datetime.datetime.now()
            #if ((time_now.minute+1)%5==0 and time_now.second<25):
            if ((time_now.minute+1)%4==0):
                time_now=round(time_now.timestamp())*1000
                df1=pd.DataFrame()
                for i in range(2*31): #2*365
                    df1=import_api(time_now-12*3600*1000*i,symbol,720,"5m").append(df1)
                df1=transform_df(df1)
                df1=preprocess_df(df1,4,24,1,15,1).reset_index(drop=True)

            beginning=-window_size-1
            df1_part=df1.iloc[beginning:]
            opens=df1_part["Open"]
            test_opens=np.asarray(df1_part["Open"][:])
            #encoded=encode(encoder,df1_part.drop(["Open","target"],axis=1).to_numpy()[:]).tolist()
            encoded=df1_part.drop(["Open","target"],axis=1).to_numpy()
            encoded=torch.tensor(encoded,device="cuda")
            seq=tensor_to_sequences(encoded[-(window_size+1):],window_size)
            test_data=tensor_to_sequences(encoded[:],window_size)
            for i in range(len(test_data)):
                    for j in range(test_data[i].shape[0]):
                        test_data[i][j]=np.array(test_data[i][j])
            df1_part["Open"].values[-1]=float(client_api.ticker_price(symbol=symbol)["price"])
            this_open=df1_part["Open"].values[-1] 

            array=[]
            preds=[]
            for j in range(net_repeating):
                array.append(net(torch.tensor([test_data[i]],device="cuda")).tolist()[0]) #using net
            array=np.asarray(array)
            for j in range(len(array[0])):
                preds.append(np.mean(array[:,j]))
            test_prediction=upscale(preds[0],0.2)*this_open #getting prediction
            buying_value=buying_threshold*this_open
            backtest_preds.append(test_prediction)
            if (buy==0 and test_prediction>buying_value): #buy when prediction > min sell value 
                buy=1
                purchase_size=round(purchase_size_usdt/this_open,4)
                client.new_order(symbol=symbol,side="BUY",type="MARKET",quantity=purchase_size,recvWindow=20000)
                desired_sell_cost=min_sell_value
                #wallet_dynamics.append(wallet-wallet_start)
                min_sell_value=min_sell_threshold*this_open
                time.sleep(0.3)

                last_trade_data["price"][0]=this_open
                last_trade_data["quantity"][0]=purchase_size
                last_trade_data["take_profit_price"][0]=min_sell_value
                last_trade_data["stop_loss_price"][0]=this_open*stop_loss
                last_trade_data.to_csv(directory+'\\last_trade_data.csv',index=False)
            if (buy==1 and (this_open<last_trade_data["stop_loss_price"][0] or this_open>last_trade_data["take_profit_price"][0])):
                client_api.new_order(symbol=symbol,side="SELL",type="MARKET",quantity=last_trade_data["quantity"][0],recvWindow=20000)
                buy=0
                loss_count=0
                last_trade_data=pd.DataFrame({'price':[0], 'quantity':[0],"take_profit_price":[0],"stop_loss_price":[0]})
                last_trade_data.to_csv(directory+'\\last_trade_data.csv',index=False)
            if (buy==1 and test_prediction<last_trade_data["price"][0]*(1-buying_threshold*loss_multiplier)):
                loss_count+=1
                if (loss_count>=loss_count_max):
                    client_api.new_order(symbol=symbol,side="SELL",type="MARKET",quantity=last_trade_data["quantity"][0],recvWindow=20000)
                    loss_count=0
                    buy=0
                    last_trade_data=pd.DataFrame({'price':[0], 'quantity':[0],"take_profit_price":[0],"stop_loss_price":[0]})
                    last_trade_data.to_csv(directory+'\\last_trade_data.csv',index=False)
            time.sleep(60)
            log_time_count+=1
            if (log_time_count==4):
                print(test_prediction)
                if(loss_count>0):
                    print("loss_count=",loss_count)
                log_time_count=0
            gc.collect()
        except Exception:
            pass


# In[ ]:


#client.new_order(symbol="ETHUSDT",side="BUY",type="LIMIT",quantity=0.02,price=1400,timeInForce="GTC")
#client.new_order(symbol="ETHUSDT",side="BUY",type="TAKE_PROFIT",quantity=0.007,stopPrice=1400)


# In[44]:


#last_trade_data=pd.DataFrame({'price':[0], 'quantity':[0],"take_profit_price":[0],"stop_loss_price":[0]})
#last_trade_data.to_csv(directory+'\\last_trade_data.csv',index=False)


# In[ ]:




