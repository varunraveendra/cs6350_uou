import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data=pd.read_csv('bank-note/train.csv',header=None)
test_data=pd.read_csv('bank-note/test.csv',header=None)

train_data[4] = np.where(train_data[4]==0, -1, 1)
test_data[4] = np.where(test_data[4]==0, -1, 1)


def affine_layer_forward(input,W,b):
  output=np.dot(input,W)+b
  cache=input,W,b
  return output, cache
  
  
def sgm_forward(input):
  output=1/(1 + np.exp(-input)) 
  cache=output
  return output,cache
  
  
def sgm(input):
  output=1/(1 + np.exp(-input)) 
  cache=output
  return output
  
def affine_layer_backward(dx,cache):
  input,W,b=cache
  #print(dx.shape,W.shape)
  try:
    dz=np.dot(dx,W.T)
  except:
    dz=np.outer(dx,W)
  dw=np.dot(input.T,dx)
  db=np.sum(dx,axis=0)
  return dz,dw,db
  

def sgm_backward(dx,cache):
  output=cache
  #print(output.shape)
  m=np.multiply(output,(1-output))
  back=np.multiply(m,dx)
  return back

def sq_loss(Y,y):
  loss=np.sum(1/2*(Y-y)**2)
  grad=(Y-y)
  return loss, grad



def three_layer_nn_train(train_data,T,hid_dim,lr,d,ran,z=0):
  if z==0:
    np.random.seed(ran)
    w1=np.random.randn(4,hid_dim-1)
    b1=np.ones(hid_dim-1)
    w2=np.random.randn(hid_dim-1,hid_dim-1)
    b2=np.ones(hid_dim-1)
    w3=np.random.randn(hid_dim-1)
    b3=1
  elif z==1:
    w1=np.zeros((4,hid_dim-1))
    b1=np.zeros(hid_dim-1)
    w2=np.zeros((hid_dim-1,hid_dim-1))
    b2=np.zeros(hid_dim-1)
    w3=np.zeros(hid_dim-1)
    b3=0
  losses=[]
  for i in range(T):
    train_data=train_data.sample(frac = 1)
    y=train_data.iloc[:,-1]
    x=train_data.iloc[:,:-1]
    x=x.to_numpy()
    y=y.to_numpy()
    gamma_t= lr/(1+(lr/d)*i)
    for j in range(len(x)):
      op1,alcache1=affine_layer_forward(x[j].reshape(1,-1),w1,b1)
      op2,scache1=sgm_forward(op1)
      op3,alcache2=affine_layer_forward(op2,w2,b2)
      op4,scache2=sgm_forward(op3)
      Y,alcache3=affine_layer_forward(op4,w3,b3)
      loss,grad=sq_loss(Y,y[j])

      dx,dw3,db3=affine_layer_backward(grad,alcache3)
      dx=sgm_backward(dx,scache2)
      dx,dw2,db2=affine_layer_backward(dx,alcache2)
      dx=sgm_backward(dx,scache1)
      dx,dw1,db1=affine_layer_backward(dx,alcache1)

      w1-=gamma_t*(dw1)
      w2-=gamma_t*(dw2)
      w3-=gamma_t*(dw3)
      b1-=gamma_t*(db1)
      b2-=gamma_t*(db2)
      b3-=gamma_t*(db3)
      losses.append(loss)

  return losses,(w1,w2,w3,b1,b2,b3)

def predict_nn(test_data,w1,w2,w3,b1,b2,b3):
  y=test_data.iloc[:,-1]
  x=test_data.iloc[:,:-1]
  x=x.to_numpy()
  y=y.to_numpy()
  op1,alcache1=affine_layer_forward(x,w1,b1)
  op2,scache1=sgm_forward(op1)
  op3,alcache2=affine_layer_forward(op2,w2,b2)
  op4,scache2=sgm_forward(op3)
  Y,alcache3=affine_layer_forward(op4,w3,b3)
  diff=np.where((np.sign(Y)*y)==-1,1,0)
  return(np.sum(diff)/diff.shape[0])

def q2a():
  x=np.array([1,1])
  y=1
  w_1=np.array([[-2,2],[-3,3]])
  w_2=np.array([[-2,2],[-3,3]])
  w_3=np.array([[2],[-1.5]])
  b_1=np.array([-1,1])
  b_2=np.array([-1,1])
  b_3=np.array([-1])
  op1,alcache1=affine_layer_forward(x.reshape(1,-1),w_1,b_1)
  op2,scache1=sgm_forward(op1)
  op3,alcache2=affine_layer_forward(op2,w_2,b_2)
  op4,scache2=sgm_forward(op3)
  Y,alcache3=affine_layer_forward(op4,w_3,b_3)
  loss,grad=sq_loss(Y,y)
  print('First Gradient:',grad)
  dx,dw3,db3=affine_layer_backward(grad,alcache3)
  dx=sgm_backward(dx,scache2)
  dx,dw2,db2=affine_layer_backward(dx,alcache2)
  dx=sgm_backward(dx,scache1)
  dx,dw1,db1=affine_layer_backward(dx,alcache1)
  print('Layer3:',dw3,db3)
  print('Layer2:',dw2,db2)
  print('Layer1:',dw1,db1)
  
def question2b(train_data,test_data):
  print('For hidden layer width: 5')
  loss,param=three_layer_nn_train(train_data,100,5,3e-4,9e-5,11)
  w1,w2,w3,b1,b2,b3=param
  print('Training error:',predict_nn(train_data,w1,w2,w3,b1,b2,b3))
  print('Test error:',predict_nn(test_data,w1,w2,w3,b1,b2,b3))
  plt.plot(loss)
  plt.xlabel('#Updates')
  plt.ylabel('Loss')
  plt.title('Loss vs Updates')
  plt.show()
  print('For hidden layer width: 10')
  loss,param=three_layer_nn_train(train_data,100,10,3e-4,1e-5,77)
  w1,w2,w3,b1,b2,b3=param
  print('Training error:',predict_nn(train_data,w1,w2,w3,b1,b2,b3))
  print('Test error:',predict_nn(test_data,w1,w2,w3,b1,b2,b3))
  plt.plot(loss)
  plt.xlabel('#Updates')
  plt.ylabel('Loss')
  plt.title('Loss vs Updates')
  plt.show()
  print('For hidden layer width: 25')
  loss,param=three_layer_nn_train(train_data,100,25,3e-4,1e-4,89)
  w1,w2,w3,b1,b2,b3=param
  print('Training error:',predict_nn(train_data,w1,w2,w3,b1,b2,b3))
  print('Test error:',predict_nn(test_data,w1,w2,w3,b1,b2,b3))
  plt.plot(loss)
  plt.xlabel('#Updates')
  plt.ylabel('Loss')
  plt.title('Loss vs Updates')
  plt.show()
  print('For hidden layer width: 50')
  loss,param=three_layer_nn_train(train_data,100,50,3e-4,1e-4,82)
  w1,w2,w3,b1,b2,b3=param
  print('Training error:',predict_nn(train_data,w1,w2,w3,b1,b2,b3))
  print('Test error:',predict_nn(test_data,w1,w2,w3,b1,b2,b3))
  plt.plot(loss)
  plt.xlabel('#Updates')
  plt.ylabel('Loss')
  plt.title('Loss vs Updates')
  plt.show()
  print('For hidden layer width: 100')
  loss,param=three_layer_nn_train(train_data,100,100,3e-4,1e-4,80)
  w1,w2,w3,b1,b2,b3=param
  print('Training error:',predict_nn(train_data,w1,w2,w3,b1,b2,b3))
  print('Test error:',predict_nn(test_data,w1,w2,w3,b1,b2,b3))
  plt.plot(loss)
  plt.xlabel('#Updates')
  plt.ylabel('Loss')
  plt.title('Loss vs Updates')
  plt.show()
  

def question2c(train_data,test_data):
  print('WHEN WEIGHTS ARE SET TO ZERO')
  print('For hidden layer width: 5')
  loss,param=three_layer_nn_train(train_data,100,5,3e-4,9e-5,11,1)
  w1,w2,w3,b1,b2,b3=param
  print('Training error:',predict_nn(train_data,w1,w2,w3,b1,b2,b3))
  print('Test error:',predict_nn(test_data,w1,w2,w3,b1,b2,b3))
  plt.plot(loss)
  plt.xlabel('#Updates')
  plt.ylabel('Loss')
  plt.title('Loss vs Updates')
  plt.show()
  print('For hidden layer width: 10')
  loss,param=three_layer_nn_train(train_data,100,10,3e-4,1e-5,77,1)
  w1,w2,w3,b1,b2,b3=param
  print('Training error:',predict_nn(train_data,w1,w2,w3,b1,b2,b3))
  print('Test error:',predict_nn(test_data,w1,w2,w3,b1,b2,b3))
  plt.plot(loss)
  plt.xlabel('#Updates')
  plt.ylabel('Loss')
  plt.title('Loss vs Updates')
  plt.show()
  print('For hidden layer width: 25')
  loss,param=three_layer_nn_train(train_data,100,25,3e-4,1e-4,89,1)
  w1,w2,w3,b1,b2,b3=param
  print('Training error:',predict_nn(train_data,w1,w2,w3,b1,b2,b3))
  print('Test error:',predict_nn(test_data,w1,w2,w3,b1,b2,b3))
  plt.plot(loss)
  plt.xlabel('#Updates')
  plt.ylabel('Loss')
  plt.title('Loss vs Updates')
  plt.show()
  print('For hidden layer width: 50')
  loss,param=three_layer_nn_train(train_data,100,50,3e-4,1e-4,82,1)
  w1,w2,w3,b1,b2,b3=param
  print('Training error:',predict_nn(train_data,w1,w2,w3,b1,b2,b3))
  print('Test error:',predict_nn(test_data,w1,w2,w3,b1,b2,b3))
  plt.plot(loss)
  plt.xlabel('#Updates')
  plt.ylabel('Loss')
  plt.title('Loss vs Updates')
  plt.show()
  print('For hidden layer width: 100')
  loss,param=three_layer_nn_train(train_data,100,100,3e-4,1e-4,80,1)
  w1,w2,w3,b1,b2,b3=param
  print('Training error:',predict_nn(train_data,w1,w2,w3,b1,b2,b3))
  print('Test error:',predict_nn(test_data,w1,w2,w3,b1,b2,b3))
  plt.plot(loss)
  plt.xlabel('#Updates')
  plt.ylabel('Loss')
  plt.title('Loss vs Updates')
  plt.show()


q2a()