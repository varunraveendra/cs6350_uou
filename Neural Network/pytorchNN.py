import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

train_data=pd.read_csv('bank-note/train.csv',header=None)
test_data=pd.read_csv('bank-note/test.csv',header=None)


train_data[4] = np.where(train_data[4]==0, -1, 1)
test_data[4] = np.where(test_data[4]==0, -1, 1)

y=train_data.iloc[:,-1]
x=train_data.iloc[:,:-1]
x_ttrain = torch.tensor(x.values, dtype=torch.float32)
y_ttrain= torch.tensor(y.values, dtype=torch.float32)
y=test_data.iloc[:,-1]
x=test_data.iloc[:,:-1]
x_ttest = torch.tensor(x.values, dtype=torch.float32)
y_ttest= torch.tensor(y.values, dtype=torch.float32)

class varyinglayersNN(nn.Module):
  def __init__(self,input_dim,output_dim,hidden_dim,depth,activation):
    super(varyinglayersNN,self).__init__()
    layers=[]
    pr_dim=input_dim
    for i in range(depth):
      layers.append(nn.Linear(pr_dim,hidden_dim))
      nn.init.xavier_uniform_(layers[-1].weight) if activation=='tanh' else nn.init.kaiming_uniform_(layers[-1].weight,nonlinearity='relu')
      layers.append(nn.Tanh()) if activation=='tanh' else layers.append(nn.ReLU())
      pr_dim=hidden_dim

    layers.append(nn.Linear(pr_dim,output_dim))
    nn.init.xavier_uniform_(layers[-1].weight) if activation=='tanh' else nn.init.kaiming_uniform_(layers[-1].weight,nonlinearity='relu')

    self.model=nn.Sequential(*layers)

  def forward(self,x):
    return self.model(x)
    
def trainloop(model,train_loader,criterion,optimizer,T):
  for i in range(T):
    for ip,l in train_loader:
      output=model(ip)
      l=l.unsqueeze(1)
      loss=criterion(output,l)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()


def testloop(model,test_loader):
  model.eval()
  pred=[]
  labe=[]
  with torch.no_grad():
    for ip,l in test_loader:
      output=model(ip)
      pred.extend(torch.sign(output).tolist())
      labe.extend(l.tolist())
  return accuracy_score(labe,pred)




train_dataset=TensorDataset(x_ttrain,y_ttrain)
train_loader=DataLoader(train_dataset,batch_size=100,shuffle=True)
test_dataset=TensorDataset(x_ttest,y_ttest)
test_loader=DataLoader(test_dataset,batch_size=100,shuffle=False)




for i in ['tanh','relu']:
  for j in [3, 5, 9]:
    for k in [5, 10, 25, 50, 100]:
      model=varyinglayersNN(4,1,k,j,i)
      optimizer=optim.Adam(model.parameters())
      criterion=nn.MSELoss()
      trainloop(model,train_loader,criterion,optimizer,100)
      print('Activation:'+i+' '+'Depth:'+ str(j)+' '+'Hidden layer:',k)
      print("Train accuracy:",testloop(model,train_loader))
      print("Test accuracy:",testloop(model,test_loader))

