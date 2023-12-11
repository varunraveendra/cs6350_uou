import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data=pd.read_csv('bank-note/train.csv',header=None)
test_data=pd.read_csv('bank-note/test.csv',header=None)

train_data[4] = np.where(train_data[4]==0, -1, 1)
test_data[4] = np.where(test_data[4]==0, -1, 1)


def sgm(input):
  output=1/(1 + np.exp(-input)) 
  cache=output
  return output

def logistic_regression(train_data,v,obj,gamma_init,d):
  w=np.zeros(train_data.iloc[0,:-1].shape[0]+1)
  loss=[]
  for j in range(100):
    train_data=train_data.sample(frac = 1)
    y=train_data.iloc[:,-1]
    x=train_data.iloc[:,:-1]
    x=x.to_numpy()
    y=y.to_numpy()
    x=np.insert(x, 0, 1, axis=1)
    gamma_t= gamma_init/(1+(gamma_init/d)*j)
    for i in range(len(x)):
      if obj=='map':
        L=len(x)*np.log(1+np.exp(-y[i]*(np.dot(w,x[i]))))+1/(2*v)*np.dot(w,w)
        gL=w/v-len(x)*y[i]*x[i]*(1-sgm(y[i]*(np.dot(w,x[i]))))
        loss.append(L)
      else:
        L=len(x)*np.log(1+np.exp(-y[i]*(np.dot(w,x[i]))))
        gL=-len(x)*y[i]*x[i]*(1-sgm(y[i]*(np.dot(w,x[i]))))
        loss.append(L)
      w-=gamma_t*gL
  return w,loss
  
  
def prediction_standard(test_data,w):
  y=test_data.iloc[:,-1]
  x=test_data.iloc[:,:-1]
  x=x.to_numpy()
  y=y.to_numpy()
  x=np.insert(x, 0, 1, axis=1)
  pred=np.dot(x,w)
  pred_s=np.sign(pred)
  diff=np.where((pred_s*y)==-1,1,0)
  return(np.sum(diff)/diff.shape[0])

def q3(train_data,test_data):
  for j in ['map','ml']:
    for i in [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]:
      w,l=logistic_regression(train_data,i,j,1e-4,1e-4)
      print(str(j).upper()+' with v=',i)
      print('Training error:',prediction_standard(train_data,w))
      print('Test error:',prediction_standard(test_data,w))
      plt.plot(l)
      plt.xlabel('#Updates')
      plt.ylabel('Loss')
      plt.title('Loss vs Updates')
      plt.show()


q3(train_data,test_data)


