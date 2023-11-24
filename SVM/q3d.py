import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data=pd.read_csv('bank-note/train.csv',header=None)
test_data=pd.read_csv('bank-note/test.csv',header=None)
train_data[4] = np.where(train_data[4]==0, -1, 1)
test_data[4] = np.where(test_data[4]==0, -1, 1)

def primal_svm(train_data,gamma_init,C,a=0,T=100):
  w=np.zeros(train_data.iloc[0,:-1].shape[0]+1)
  for j in range(T):
    train_data=train_data.sample(frac = 1)
    y=train_data.iloc[:,-1]
    x=train_data.iloc[:,:-1]
    x=x.to_numpy()
    y=y.to_numpy()
    x=np.insert(x, 0, 1, axis=1)
    gamma_t= gamma_init/(1+(gamma_init/a)*j) if a!=0 else gamma_init/(1+j)
    for i in range(len(x)):
      w0 = np.copy(w)
      w0[0] = 0
      if (y[i]*(np.dot(w,x[i])))<=1:
        w+=(-gamma_t*w0)+(gamma_t*C*len(x)*y[i]*x[i])
      else:
        w+=-gamma_t*w0
  return w
  

def gaussian_kernel(x,z,g):
  tem=np.zeros((x.shape[0],z.shape[0]))
  for i in range(x.shape[0]):
    for j in range(z.shape[0]):
      tem[i,j]=np.linalg.norm(x[i]-z[j])
  tem=np.exp(np.square(tem)/(-g))
  return tem


def dual_perceptron(train_data,g,T=100):
  y=train_data.iloc[:,-1]
  x=train_data.iloc[:,:-1]
  x=x.to_numpy()
  y=y.to_numpy()
  x=np.insert(x, 0, 1, axis=1)
  c=np.zeros_like(y)
  for j in range(T):
    for i in range(len(x)):
      gk=gaussian_kernel(x,x[i:i+1],g)
      gk=gk.reshape(gk.shape[0])
      #print((np.sum(c*y*gk)))
      if y[i]*(np.sum(c*y*gk))<=0:
        #print('kk')
        c[i]+=1
  return c


def prediction_perceptron_dual(train_data,test_data,c,g):
  yt=train_data.iloc[:,-1]
  xt=train_data.iloc[:,:-1]
  xt=xt.to_numpy()
  yt=yt.to_numpy()
  xt=np.insert(xt, 0, 1, axis=1)
  y=test_data.iloc[:,-1]
  x=test_data.iloc[:,:-1]
  x=x.to_numpy()
  y=y.to_numpy()
  x=np.insert(x, 0, 1, axis=1)
  pred=np.zeros_like(y)
  for i in range(len(x)):
      gk=gaussian_kernel(xt,x[i:i+1],g)
      gk=gk.reshape(gk.shape[0])
      pred[i]=np.sign(np.sum(c*yt*gk))
  pred_s=pred
  diff=np.where((pred_s*y)==-1,1,0)
  return(np.sum(diff)/diff.shape[0])



def prediction_gaussian(train_data,test_data,w,g,b=0):
  yt=train_data.iloc[:,-1]
  xt=train_data.iloc[:,:-1]
  xt=xt.to_numpy()
  yt=yt.to_numpy()
  y=test_data.iloc[:,-1]
  x=test_data.iloc[:,:-1]
  x=x.to_numpy()
  y=y.to_numpy()
  pred=np.dot(w*yt,gaussian_kernel(xt,x,g))+b
  pred_s=np.sign(pred)
  diff=np.where((pred_s*y)==-1,1,0)
  return(np.sum(diff)/diff.shape[0])


def dual_svm(train_data,C,g=0):
    y=train_data.iloc[:,-1]
    x=train_data.iloc[:,:-1]
    x=x.to_numpy()
    y=y.to_numpy()
    yy=np.outer(y,y)
    xx=gaussian_kernel(x,x,g) if g != 0 else np.inner(x,x)
    yyxx=yy*xx
    dual= lambda al: (0.5*np.sum(yyxx*np.outer(al,al)))-np.sum(al)
    cons = ({'type': 'eq', 'fun': lambda al: np.dot(al,y)})
    bnds=[(0., C)] * y.shape[0]
    res=minimize(dual,np.full(y.shape[0],0.),method='SLSQP',constraints=cons,bounds=bnds)
    opti_w= 0 if g!=0 else np.dot(res.x*y,x)
    s_vec=x[res.x>0]
    y_vec=y[res.x>0]
    opti_b=np.mean(y_vec-np.dot(res.x*y,gaussian_kernel(x,s_vec,g))) if g!=0 else np.mean(y_vec-np.dot(s_vec,opti_w))
    opti_wb= 0 if g!=0 else np.insert(opti_w, 0, opti_b)
    return res,opti_wb,opti_b


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
 
 
 
  
def question2(train_data,test_data):
  c=[100/873,500/873,700/873]
  print('FIRST WEIGHT IS BIAS')
  print("Learning Rate setting 1:")
  for i in c:
    wsvm=primal_svm(train_data,1e-3,i,1e-5)
    print('For C='+str(i)+' the parameters learnt are',wsvm)
    print('Training Error=',prediction_standard(train_data,wsvm))
    print('Test Error=',prediction_standard(test_data,wsvm))
  print('--------------------------------------------------------------------------------------------------------')
  print("Learning Rate setting 2:")
  for i in c:
    wsvm=primal_svm(train_data,1e-3,i)
    print('For C='+str(i)+' the parameters learnt are',wsvm)
    print('Training Error=',prediction_standard(train_data,wsvm))
    print('Test Error=',prediction_standard(test_data,wsvm))

def question3a(train_data,test_data):
  c=[100/873,500/873,700/873]
  print('--------------------------------------------------------------------------------------------------------')
  print("Linear SVM:")
  print('FIRST WEIGHT IS BIAS')
  print('Running optimizations ......(ignore warnings).......')
  for i in c:
    _,opti_wb,_=dual_svm(train_data,i)
    print('For C='+str(i)+' the parameters learnt are',opti_wb)
    print('Training Error=',prediction_standard(train_data,opti_wb))
    print('Test Error=',prediction_standard(test_data,opti_wb))


def question3bc(train_data,test_data):
  c=[100/873,500/873,700/873]
  gamma=[0.1,0.5,1,5,100]
  print('--------------------------------------------------------------------------------------------------------')
  print("Non-Linear SVM:")
  print('Running optimizations ......(ignore warnings).......')
  for i in c:
      print('For C=',i)
      for j in gamma:
        al,_,b=dual_svm(train_data,i,j)
        print('gamma=',j)
        print('Number of Support Vectors=',np.sum([al.x>0]))
        print('Training Error=',prediction_gaussian(train_data,train_data,al.x,j,b))
        print('Test Error=',prediction_gaussian(train_data,test_data,al.x,j,b))


def question3c(train_data,test_data):
  c=500/873
  gamma=[0.1,0.5,1,5,100]
  print('--------------------------------------------------------------------------------------------------------')
  print("Non-Linear SVM: with c= 500/873")
  print('Running optimizations ......(ignore warnings).......')
  al1,_,_=dual_svm(train_data,c,gamma[0])
  al2,_,_=dual_svm(train_data,c,gamma[1])
  al3,_,_=dual_svm(train_data,c,gamma[2])
  al4,_,_=dual_svm(train_data,c,gamma[3])
  al5,_,_=dual_svm(train_data,c,gamma[4])
  print("For gamma 0.1 and 0.5 ",np.sum([al1.x*al2.x>0]))
  print("For gamma 0.5 and 1 ",np.sum([al2.x*al3.x>0]))
  print("For gamma 1 and 5 ",np.sum([al3.x*al4.x>0]))
  print("For gamma 5 and 100 ",np.sum([al4.x*al5.x>0]))



def question3d(train_data,test_data):
  gamma=[0.1,0.5,1,5,100]
  print('Dual perceptron:')
  for i in gamma:
    c=dual_perceptron(train_data,i,10)
    print('For gamma=',i)
    print('Training Error=',prediction_perceptron_dual(train_data,train_data,c,i))
    print('Test Error=',prediction_perceptron_dual(train_data,test_data,c,i))
    
    
    
#question2(train_data,test_data)
#question3a(train_data,test_data)
#question3bc(train_data,test_data)
#question3c(train_data,test_data)
question3d(train_data,test_data)
