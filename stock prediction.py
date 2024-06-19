import torch 
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv(r"E:\stock prediction using pytorch\Tesla.csv - Tesla.csv.csv")
data=data['High'].values
data=np.array(data,dtype=np.float32).reshape(-1,1)

ms=MinMaxScaler(feature_range=(-1,1))
data=ms.fit_transform(data)

x=[]
y=[]
l=3
for i in range(len(data)-l-1):
    x.append([data[i:l]])
    y.append(data[l])
    l+=1

x=torch.tensor(x,dtype=torch.float32).squeeze(3)
y=torch.tensor(y,dtype=torch.float32)


class my_data(Dataset):

    def __init__(self,x,y):

        self.x=x
        self.y=y
        self.n=x.shape[0]
    def __getitem__(self, index):
        return x[index],y[index]
    
    def __len__(self):
        return self.n
dataset=my_data(x,y)

batch_data=DataLoader(dataset,batch_size=8,drop_last=True)

    
class sp(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1=nn.LSTM(3,128,num_layers=2)
        self.d1=nn.Linear(128,500)
        self.a1=nn.ReLU()
        self.d2=nn.Linear(500,100)
        self.a2=nn.ReLU()
        self.d3=nn.Linear(100,1)

    def forward(self,x):
        self.r=self.l1(x)[0]
        return self.d3(self.a2(self.d2(self.a1(self.d1(self.r)))))
    
model=sp().to("cuda")

opt=torch.optim.Adam(params=model.parameters())
loss=nn.MSELoss()

for i in range(1000):
    TOTAL_LOSS=0
    for q,(j,k) in enumerate(batch_data):
        yp=model(j.to("cuda"))
        # print(k.shape)
        diff=loss(yp,k.to("cuda"))

        opt.zero_grad()
        diff.backward()
        opt.step()
        TOTAL_LOSS+=diff

    print(f"the epoch {i} with loss {TOTAL_LOSS/52}")



ypred=model(x.to("cuda"))
ypred=ypred.cpu().detach().numpy()


from matplotlib import pyplot as plt
plt.plot(data[4:],c="r")
plt.plot(ypred.squeeze(1),c='b')
plt.show()

# ypred=ms.inverse_transform(ypred.squeeze().reshape((-1,1)))