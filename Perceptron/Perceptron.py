import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data=pd.read_csv('bank-note/train.csv',header=None)
test_data=pd.read_csv('bank-note/test.csv',header=None)
train_data[4] = np.where(train_data[4]==0, -1, 1)
test_data[4] = np.where(test_data[4]==0, -1, 1)

def standard_perceptron(train_data,r,T=10):
  w=np.zeros(train_data.iloc[0,:-1].shape[0]+1)
  for j in range(T):
    train_data=train_data.sample(frac = 1)
    y=train_data.iloc[:,-1]
    x=train_data.iloc[:,:-1]
    x=x.to_numpy()
    y=y.to_numpy()
    x=np.insert(x, 0, 1, axis=1)
    for i in range(len(x)):
      if (y[i]*(np.dot(w,x[i])))<=0:
        w+=r*y[i]*x[i]
  return w

def voted_perceptron(train_data,r,T=10):
  w=np.zeros(train_data.iloc[0,:-1].shape[0]+1)
  bag=[]
  cm=0
  for j in range(T):
    train_data=train_data.sample(frac = 1)
    y=train_data.iloc[:,-1]
    x=train_data.iloc[:,:-1]
    x=x.to_numpy()
    y=y.to_numpy()
    x=np.insert(x, 0, 1, axis=1)
    for i in range(len(x)):
      if (y[i]*(np.dot(w,x[i])))<=0:
        bag.append((w.copy(),cm))
        w+=r*y[i]*x[i]
        cm=1
      else:
        cm+=1
  bag.append((w.copy(),cm))
  return bag

def averaged_perceptron(train_data,r,T=10):
  w=np.zeros(train_data.iloc[0,:-1].shape[0]+1)
  a=np.zeros(train_data.iloc[0,:-1].shape[0]+1)
  for j in range(T):
    train_data=train_data.sample(frac = 1)
    y=train_data.iloc[:,-1]
    x=train_data.iloc[:,:-1]
    x=x.to_numpy()
    y=y.to_numpy()
    x=np.insert(x, 0, 1, axis=1)
    for i in range(len(x)):
      if (y[i]*(np.dot(w,x[i])))<=0:
        w+=r*y[i]*x[i]
      a+=w
  return a
  

def prediction_standard_averaged(test_data,w):
  y=test_data.iloc[:,-1]
  x=test_data.iloc[:,:-1]
  x=x.to_numpy()
  y=y.to_numpy()
  x=np.insert(x, 0, 1, axis=1)
  pred=np.dot(x,w)
  pred_s=np.sign(pred)
  diff=np.where((pred_s*y)==-1,1,0)
  return(np.sum(diff)/diff.shape[0])
  
  
def prediction_voted(test_data,bag):
  y=test_data.iloc[:,-1]
  x=test_data.iloc[:,:-1]
  x=x.to_numpy()
  y=y.to_numpy()
  pred_tot=np.zeros_like(y)
  x=np.insert(x, 0, 1, axis=1)
  for i in bag:
    w=i[0]
    c=i[1]
    pred=np.dot(x,w)
    pred_c=c*np.sign(pred)
    pred_tot=np.add(pred_tot,pred_c)
  pred_s=np.sign(pred_tot)
  diff=np.where((pred_s*y)==-1,1,0)
  return(np.sum(diff)/diff.shape[0])
  
  
w1=standard_perceptron(train_data,1e-5)

w2=voted_perceptron(train_data,1e-5)

w3=averaged_perceptron(train_data,1e-5)

print(w1,'Standard Perceptron Weights')
print(w2,'Voted Perceptron Weights')
print(w3,'Averaged Perceptron Weights')

print(prediction_standard_averaged(test_data,w1),'Standard test error')
print(prediction_standard_averaged(test_data,w3),'Averaged test error')
print(prediction_voted(test_data,w2),'Voted test error')

