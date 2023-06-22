#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import seaborn as sns


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# In[2]:


def import_api(time_last,symbol="BTCUSDT",limit=12*60):
    params={"symbol":symbol,"limit":limit,"interval":"1m","endTime":time_last,"startTime":(time_last-12*3600*1000)}
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

def df_to_X_y(df, window_size=5,seq=1):
    df_as_np = df.to_numpy()
    X = []
    y = []
    if (seq):
        for i in range(len(df_as_np)-window_size):
            row = [a for a in (df.iloc[i:i+window_size].drop("target",axis=1).values)]
            X.append(row)
            label = df["target"][i+window_size-1]
            y.append(label)
    else:
        for i in range(len(df_as_np)):
            row = df.iloc[i].drop("target").values
            X.append(row)
            label = df["target"][i]
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
        label = data[data.columns[-1]][i]
        y.append(label)
    return np.array(X), np.array(y)
def preprocess_df(df,target_range,hours,hours_interval,days,days_interval,scaling_range=0.2,scaling=1):
    target=[]
    mean_price=[]
    open_delta=[]
    prev_price=[]
    concat_hours=3
    concat_days=8
    window=target_range
    #for i in range(df["Open"].size-window):
    #    target.append(df["Open"][i+window])
    for i in range(df["Open"].size-window):
        target.append((df["High"][i+window]+df["Low"][i+window])/2)
        mean_price.append((df["High"][i]+df["Low"][i])/2)
    df=df[0:df["Open"].size-window]
    df["mean_price"]=mean_price
    df["target"]=target
    df=df.drop(["Close_time","Taker_buy_base_asset_volume","Volume","Close"],axis=1)#,"Low","High","Number_of_trades"],axis=1)
    for i in range(1,hours//hours_interval+1):
        df["mean_"+str(i*hours_interval)+"_hours"]=moving_average(df["Open"],i*(12//concat_hours)*hours_interval,concat_hours)
    for i in range(1,days//days_interval+1):
        df["mean_"+str(i*days_interval)+"_days"]=moving_average(df["Open"],i*12*(24//concat_days)*days_interval,concat_days)
    drop_col=["Open",'Open_time','weekday',"Number_of_trades","mean_price"]
    df["weekday"]=df["weekday"].astype(float)
    df["Number_of_trades"]=df["Number_of_trades"].astype(float)
    for i in range(1,df["Open"].size):
        df["Open_time"][i]/=86400
        df["weekday"][i]/=6.0
        df["Number_of_trades"][i]/=10000
        delta=df["mean_price"][i]-df["mean_price"][i-1]
        prev_price.append(df["mean_price"][i-1])
        open_delta.append(delta)
    df=df.drop(0,axis=0)
    #df["mean_price_delta"]=open_delta
    df["prev_price"]=prev_price
    for col in df.drop(drop_col,axis=1).columns:
        for i in range(1,df["Open"].size+1):
            #df[col][i]=df[col][i]/df["Open"][i]
            df[col][i]=df[col][i]/df["mean_price"][i]
            pass
    cols=df.drop(drop_col,axis=1).columns
    if scaling:
        for i in cols:
            for j in range(1,df["Open"].size+1):
                df[i][j]=(df[i][j]-(1-scaling_range))/(scaling_range*2)
    return df

def upscale(input_data,scaling_range):
    return input_data*2*scaling_range-scaling_range+1


# In[3]:


def show_features_heatmap(df):
    corr=df.corr()
    result=sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    return result


# In[ ]:





# In[4]:


class autoencoder(nn.Module):
    def __init__(self,drop,hidden_size,test_size):
        super(autoencoder, self).__init__()
        self.norm=nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(41,hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc3 = nn.Linear(hidden_size, test_size,bias=True)
        self.fc4 = nn.Linear(test_size, hidden_size,bias=True)
        self.fc5 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc6 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc7 = nn.Linear(hidden_size, 41,bias=True)
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


# In[ ]:





# In[5]:


#RELUS

class Net_r(nn.Module):
    def __init__(self,drop,hidden_size,input_size):
        super(Net_r, self).__init__()
        self.lstm_active=0
        self.norm=nn.BatchNorm1d(hidden_size)
        self.lstm = nn.LSTM(num_layers=1,input_size=input_size, hidden_size=hidden_size,batch_first=True)
        self.fc1 = nn.Linear(input_size, hidden_size,bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc4 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc5 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc6 = nn.Linear(hidden_size, 1,bias=True)
        self.dropout = nn.Dropout(drop)
    def forward(self, x,train):
        if (self.lstm_active):
            x,_ = self.lstm(x)
            x=x[:,-1,:]
        else:
            x=self.fc1(x)
        #x=self.norm(x)
        x=F.relu(x)
        if train:
            #x=self.norm(x)
            pass
        
        x=self.fc2(x)
        x=F.relu(x)
        
        x=self.fc3(x)
        x=F.relu(x)
        
        if train:
            x = self.dropout(x)
        
        x=self.fc4(x)
        x=F.tanh(x)
        
        x=self.fc5(x)
        x=F.relu(x)
        
        if train:
            x = self.dropout(x)
        
        x = self.fc6(x)
        return x


# In[6]:


#LSTM##

class Net(nn.Module):
    def __init__(self,drop,hidden_size,input_size):
        super(Net, self).__init__()
        self.lstm_active=0
        #self.norm=nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.LSTM(num_layers=1,input_size=input_size, hidden_size=hidden_size,batch_first=True)#lstm input
        self.fc2 = nn.Linear(input_size, hidden_size//2,bias=True)#standart input
        self.fc3 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc4 = nn.Linear(hidden_size, hidden_size,bias=True)
        self.fc5 = nn.Linear(hidden_size//2, hidden_size//4,bias=True)
        self.fc6 = nn.Linear(hidden_size//4, 1,bias=True)
        #self.dropout = nn.Dropout(drop)
    def forward(self, x):
        if (self.lstm_active):
            x,_ = self.fc1(x)
            x=x[:,-1,:]
        else:
            x=self.fc2(x)
        
        #x=F.tanh(x)
        #x = self.fc3(x)
        x=F.tanh(x)
        #x = F.logsigmoid(x)
        #x=F.relu(x)
        #x = F.logsigmoid(self.fc4(x))
        x = F.relu(self.fc5(x))
        #x = self.dropout(x)
        x = self.fc6(x)
        return x


# In[7]:


def mae_func(pred_data,real_data):
    mae=0
    for i in range(len(pred_data)):
        mae+=abs(pred_data[i]-real_data[i])
    mae/=len(pred_data)
    return mae


def train(learning_rate,
          batch_size,epochs,
          momentum,train_data,test_data,X_train_opens,X_test_opens,
          net,net_max_std,net_min_err_train,net_min_err_test,
          decay=0,min_error=0,direction_punish=0,direction_reward=0,scaling=1,directory="streamlit_data"):
    loss_train_single=[]
    loss_test_single=[]
    stds_single=[]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaling_range=0.2
    X_train=train_data[0]
    X_test=test_data[0]
    y_train=train_data[1]
    y_test=test_data[1]
    
    if (train_data[0].shape[0]>2000):
        parting=X_train.shape[0]//500
        X_test_part=X_test[::parting]
        X_train_part=X_train[::parting*10]
        y_test_part=y_test[::parting]
        y_train_part=y_train[::parting*10]
        X_train_part_opens=X_train_opens[::parting*10]
        X_test_part_opens=X_test_opens[::parting]
    else:
        parting=X_train.shape[0]//1#//500
        X_test_part=X_test#[::parting]
        X_train_part=X_train#[::parting*10]
        y_test_part=y_test#[::parting]
        y_train_part=y_train#[::parting*10]
        X_train_part_opens=X_train_opens
        X_test_part_opens=X_test_opens
    train_batches=len(X_train)//batch_size
    last_batch=len(X_train)%batch_size+1
    #net = Net(drop,hidden_size,X_train.shape[-1]).to(device).double()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum,weight_decay=decay)
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=decay)
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    train_loss=nn.L1Loss()
    loss_train=[]
    loss_test=[]
    y_test_part_scaled=torch.tensor(y_test_part,device="cuda")
    y_train_part_scaled=torch.tensor(y_train_part,device="cuda")
    
    if scaling:
        y_test_part_scaled=upscale(y_test_part_scaled,scaling_range)*X_test_part_opens
        y_train_part_scaled=upscale(y_train_part_scaled,scaling_range)*X_train_part_opens
    for epoch in range(epochs):
        for batch in range(train_batches+1):
            if (batch<train_batches):
                X_train_batch=torch.tensor(X_train[batch*batch_size:(batch_size*(batch+1))],device="cuda")
                X_opens_batch=X_train_opens[batch*batch_size:(batch_size*(batch+1))]
                y_train_batch=torch.tensor(y_train[batch*batch_size:(batch_size*(batch+1))],device="cuda")
            else:
                X_train_batch=torch.tensor(X_train[batch*batch_size:batch*batch_size+last_batch],device="cuda")
                X_opens_batch=X_train_opens[batch*batch_size:batch*batch_size+last_batch]
                y_train_batch=torch.tensor(y_train[batch*batch_size:batch*batch_size+last_batch],device="cuda")
            #y_targets=upscale(y_train_batch,scaling_range)*X_opens_batch
            net_out=net(X_train_batch).reshape(1,-1)[0]#*X_opens_batch
            #net_out=net.predict(X_train_batch).reshape(1,-1)[0]#*X_opens_batch
            loss = criterion(net_out,y_train_batch)#,y_targets)
            #loss.requires_grad=True
            #loss = torch.sqrt(criterion(net_out, torch.tensor(y_train_batch,device="cuda")))
            if scaling:
                for i in range(net_out.shape[0]):
                    if((net_out[i]-0.5)*(y_train_batch[i]-0.5)<0):
                        loss*=(1+direction_punish)
                    else:
                        loss*=(1-direction_reward)
            #optimizer.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #with torch.no_grad():
        
        #net_out_test=net(torch.tensor(X_test_part,device="cuda"))
        #net_out_test=net.predict(torch.tensor(X_test_part,device="cuda")).detach().reshape(1,-1)[0]
        #net_out_train=net.predict(torch.tensor(X_train_part,device="cuda")).detach().reshape(1,-1)[0]
        net_out_test=net(torch.tensor(X_test_part,device="cuda").detach()).reshape(1,-1)[0]
        net_out_train=net(torch.tensor(X_train_part,device="cuda").detach()).reshape(1,-1)[0]
        if scaling:
            net_out_train=upscale(net_out_train,scaling_range)*X_train_part_opens
            net_out_test=upscale(net_out_test,scaling_range)*X_test_part_opens
            #print(net_out_test[:20])
            #mae=mae_func(net_out_test,y_test_part_scaled)
            #loss_test.append(train_loss(net_out_test,y_test_part_scaled))#.item())
            loss_test_single.append(train_loss(net_out_test,torch.tensor(y_test_part_scaled,device="cuda")).item())#(mae.item())
            #net_out_train=net(torch.tensor(X_train_part,device="cuda"))
            #loss_train.append(train_loss(net_out_train,y_train_part_scaled))#.item())
            loss_train_single.append(train_loss(net_out_train,torch.tensor(y_train_part_scaled,device="cuda")).item())
        else:
            #loss_test.append(train_loss(net_out_test,y_test_part))#.item())
            loss_test_single.append(train_loss(net_out_test,torch.tensor(y_test_part,device="cuda")).item())#(mae.item())
            #net_out_train=net(torch.tensor(X_train_part,device="cuda"))
            #loss_train.append(train_loss(net_out_train,y_train_part))#.item())
            loss_train_single.append(train_loss(net_out_train,torch.tensor(y_train_part,device="cuda")).item())
            
            gc.collect()
        #this_std=np.std(net.predict(torch.tensor(X_test[::70],device="cuda") ).tolist())
        #this_std=np.std(net.predict(torch.tensor(X_test,device="cuda") ).tolist())
        this_std=np.std(net(torch.tensor(X_test,device="cuda") ).tolist())
        stds_single.append(this_std)
        if (epoch==0):
            max_std=0
            min_err_test=loss_test_single[-1]
            min_err_train=loss_train_single[-1]
            
            
        if (this_std>max_std):
            path_std=os.path.join(directory,"std_net.pth")
            torch.save(net.state_dict(), path_std)
            net_max_std.load_state_dict(torch.load(path_std))
            max_std=this_std
        if (loss_train_single[-1]<min_err_train):
            path_min_err_train=os.path.join(directory,"err_train_net.pth")
            torch.save(net.state_dict(), path_min_err_train)
            net_min_err_train.load_state_dict(torch.load(path_min_err_train))
            min_err_train=loss_train_single[-1]
        if (loss_test_single[-1]<min_err_test):
            path_min_err_test=os.path.join(directory,"err_test_net.pth")
            torch.save(net.state_dict(), path_min_err_test)
            net_min_err_test.load_state_dict(torch.load(path_min_err_test))
            min_err_test=loss_test_single[-1]
        if (epoch%5==1):
            #print("train_loss: "+str(loss_train_single[-1])+ " / test loss: "+str(loss_test_single[-1])+" / std: "+str(this_std))
            print('train_loss: %.5f / test loss: %.5f / std: %.5f'% (loss_train_single[-1],loss_test_single[-1], this_std) )
        #if (loss_test_single[-1]<min_error):
        if (loss_train_single[-1]<min_error):
            break
        
    return loss_train_single,loss_test_single,stds_single


# In[8]:


def prepare_data_for_train(file,encoder=0,lstm=0, window_size=30,scaling=1,device="cuda"):
    scaling_range=0.2
    df=pd.read_csv(file).reset_index(drop=True)
    df=preprocess_df(df,6,4,1,21,4,scaling=scaling,scaling_range=scaling_range).reset_index(drop=True)
    if (encoder==0 and lstm):
        X,y=df_to_X_y(df,window_size) #for no enc
    if (encoder==0 and lstm==0):
        X,y=df_to_X_y(df,window_size,0) #no lstm no enc
    if encoder:
        X,y=tensor_to_X_y(arr,window_size) #for enc
        encoder=autoencoder(0,64,10).cuda().double()
        encoder.load_state_dict(torch.load(directory+'\\encoder_05-07-2022_22-03-48_64_10.pth'))
        encoder.eval()


    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                            test_size = 0.10,
                                                            random_state =123,
                                                            shuffle=False)

    train_idxes=sklearn.utils.shuffle(range(len(X_train)))
    X_train_shuffled=[]
    y_train_shuffled=[]
    for i in train_idxes:
        X_train_shuffled.append(X_train[i])
        y_train_shuffled.append(y_train[i])
    X_train=np.array(X_train_shuffled[:500])
    y_train=np.array(y_train_shuffled[:500])
    del([X_train_shuffled,y_train_shuffled,train_idxes])


    if (encoder==0 and lstm==0):
        X_train_opens=torch.tensor(X_train[:,1],device=device) #for df_to_x_y
        X_test_opens=torch.tensor(X_test[:,1],device=device)
        if scaling:
            X_train=np.concatenate([X_train[:,2:],X_train[:,[0]]],axis=1)
            X_test=np.concatenate([X_test[:,2:],X_test[:,[0]]],axis=1)

    if (encoder==0 and lstm==1):
        X_train_opens=torch.tensor(X_train[:,:,1][:,window_size-1],device=device) #for df_to_x_y, lstm
        X_test_opens=torch.tensor(X_test[:,:,1][:,window_size-1],device=device)
        if scaling:
            X_train=np.concatenate([X_train[:,:,2:],X_train[:,:,[0]]],axis=2)
            X_test=np.concatenate([X_test[:,:,2:],X_test[:,:,[0]]],axis=2)

    if(encoder==1):
        X_train_opens=torch.tensor(X_train[:,:,-1][:,window_size-1],device=device) #for tensor_to_x_y
        X_test_opens=torch.tensor(X_test[:,:,-1][:,window_size-1],device=device)
        if scaling:
            X_train=X_train[:,:,:-1]
            X_test=X_test[:,:,:-1]

    return X_train, X_test, y_train, y_test, X_train_opens, X_test_opens
    #LOAD NET
    #name="\\net_24-10-2022_18-33-11_16_128"
    #path=directory+name
    #net=Net(0.6,128,16).cuda().double()
    #net.load_state_dict(torch.load(path+".pth"))
    #net.train()


# In[22]:


def train_bot(X_train, X_test, y_train, y_test,X_train_opens,X_test_opens,epochs=1000,lr=0.001,hidden_size=512,drop=0,batch_sz=25,normalized_input=0,decay=0,momentum=0.8,grid_search=0,scaling=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.empty_cache() 
    X_train_normalized=torch.nn.functional.normalize(torch.tensor(X_train,device="cuda"),2,0)
    X_test_normalized=torch.nn.functional.normalize(torch.tensor(X_test,device="cuda"),2,0)
    net = Net(drop,hidden_size,X_train.shape[-1]).to(device).double()
    runs_count=0
    net_max_std=Net(drop,hidden_size,X_train.shape[-1]).to(device).double()
    net_min_err_train=Net(drop,hidden_size,X_train.shape[-1]).to(device).double()
    net_min_err_test=Net(drop,hidden_size,X_train.shape[-1]).to(device).double()
    
    min_err=0.0007
    runs_count+=1

   
    if (grid_search):
        min_loss_train_arr=[]
        min_loss_test_arr=[]
        for hidden_size_idx in range(1,10):
            hidden_size=hidden_size_idx*64
            net = Net(drop,hidden_size,X_train.shape[-1]).to(device).double()
            net_max_std=Net(drop,hidden_size,X_train.shape[-1]).to(device).double()
            net_min_err_train=Net(drop,hidden_size,X_train.shape[-1]).to(device).double()
            net_min_err_test=Net(drop,hidden_size,X_train.shape[-1]).to(device).double()
            for batch_sz_idx in range(1,2):
                for lr_idx in range (1,4):
                    loss_train_single,loss_test_single,stds_single=train(net=net,net_max_std=net_max_std,net_min_err_train=net_min_err_train,net_min_err_test=net_min_err_test,
                                                                         learning_rate=0.05/(10*lr_idx), 
                                                                         batch_size=50-batch_sz_idx*10, epochs=1000, momentum=momentum,
                                                                         decay=decay,train_data=[X_train,y_train],test_data=[X_test,y_test],
                                                                         X_train_opens=X_train_opens,X_test_opens=X_test_opens,min_error=min_err)
            min_loss_train_arr.append(np.asarray(net_min_err_train(torch.tensor(X_test,device="cuda")).tolist()))
            min_loss_test_arr.append(np.asarray(net_min_err_test(torch.tensor(X_test,device="cuda")).tolist()))   
    else:
        if normalized_input:
            loss_train_single,loss_test_single,stds_single=train(net=net,net_max_std=net_max_std,net_min_err_train=net_min_err_train,net_min_err_test=net_min_err_test,
                  learning_rate=lr, batch_size=batch_sz, epochs=epochs, momentum=momentum,decay=decay,
                  train_data=[X_train_normalized,y_train],test_data=[X_test_normalized,y_test],X_train_opens=X_train_opens,X_test_opens=X_test_opens,min_error=min_err)#,direction_punish=0.1,direction_reward=0.1)
        else:
            loss_train_single,loss_test_single,stds_single=train(net=net,net_max_std=net_max_std,net_min_err_train=net_min_err_train,net_min_err_test=net_min_err_test,
                  learning_rate=lr, batch_size=batch_sz, epochs=epochs, momentum=momentum,decay=decay,
                  train_data=[X_train,y_train],test_data=[X_test,y_test],X_train_opens=X_train_opens,X_test_opens=X_test_opens,min_error=min_err)
        err_df=pd.DataFrame({"train":loss_train_single,"test":loss_test_single})
        return net,net_min_err_train,net_min_err_test,net_max_std,err_df,stds_single
    return net,net_min_err_train,net_min_err_test,net_max_std,min_loss_train_arr,min_loss_test_arr
        


# In[ ]:





# In[13]:


def data_viz_1():
    if normalized_input:
        preds=pd.DataFrame(net(torch.tensor(X_test_normalized,device="cuda")).tolist())
        array=np.asarray(net_max_std(torch.tensor(X_test_normalized,device="cuda")).tolist())
        min_train=np.asarray(net_min_err_train(torch.tensor(X_train_normalized,device="cuda")).tolist())
    else:
        preds=pd.DataFrame(net(torch.tensor(X_test,device="cuda")).tolist())
        array=np.asarray(net_max_std(torch.tensor(X_test,device="cuda")).tolist())
        min_train=np.asarray(net_min_err_train(torch.tensor(X_test,device="cuda")).tolist())

    #data=pd.DataFrame(y_test[::70][:-15])
    data=pd.DataFrame({"targets":y_test})
    data["preds"]=preds
    data["max_std_preds"]=array
    data["min_train"]=min_train
    #data["0.5"]=pd.DataFrame([0.5 for i in range(preds.size)])
    data.plot()


# In[14]:


def data_viz_2():
    if X_train.shape[0]>1000:
        if normalized_input:
            preds=pd.DataFrame(net(torch.tensor(X_train_normalized[::100][-15:],device="cuda")).tolist())
            array=np.asarray(net_max_std(torch.tensor(X_train_normalized[::100][-15:],device="cuda")).tolist())
            min_train=np.asarray(net_min_err_train(torch.tensor(X_train_normalized[::100][-15:],device="cuda")).tolist())
        else:
            preds=pd.DataFrame(net(torch.tensor(X_train[::100][-15:],device="cuda")).tolist())
            array=np.asarray(net_max_std(torch.tensor(X_train[::100][-15:],device="cuda")).tolist())
            min_train=np.asarray(net_min_err_train(torch.tensor(X_train[::100][-15:],device="cuda")).tolist())
        data=pd.DataFrame({"targets":y_train[::100][-15:]})
    else:
        if normalized_input:
            preds=pd.DataFrame(net(torch.tensor(X_train_normalized,device="cuda")).tolist())
            array=np.asarray(net_max_std(torch.tensor(X_train_normalized,device="cuda")).tolist())
            min_train=np.asarray(net_min_err_train(torch.tensor(X_train_normalized,device="cuda")).tolist())
        else:
            preds=pd.DataFrame(net(torch.tensor(X_train,device="cuda")).tolist())
            array=np.asarray(net_max_std(torch.tensor(X_train,device="cuda")).tolist())
            min_train=np.asarray(net_min_err_train(torch.tensor(X_train,device="cuda")).tolist())
        data=pd.DataFrame({"targets":y_train})


# In[15]:


def save_net(net,path):
    torch.save(net.state_dict(), path)


# In[ ]:




