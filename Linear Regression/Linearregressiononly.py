import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('concrete+slump+test/data.csv')
data=data.drop(columns=data.columns[-2:],axis=1)
data=data.drop(columns='No',axis=1)

train_data=data.iloc[:53]
test_data=data.iloc[53:]
test_data=test_data.reset_index(drop=True)

def batchgradientdecent(train_data,r,T,W):
  #vectorized
  y=train_data.iloc[:,-1]
  x=train_data.iloc[:,:-1]
  x=x.to_numpy()
  y=y.to_numpy()
  x=np.insert(x, 0, 1, axis=1)
  w=W
  losses=[]
  iter=[]
  for i in range(T):
    #output.clear()
    iter.append(i)
    diff=y-np.dot(x,w)
    #print(diff.shape)
    diffs=diff**2
    loss=np.sum(diffs)/2
    losses.append(loss)
    dw=-np.dot(diff.T,x)
    t=w
    w=w-r*dw
    if np.linalg.norm(t-w)<0.0000001:
      return w,losses,iter,i
    print(loss)
  return w,losses,iter,i

k=np.zeros(8)
w,l,i,j=batchgradientdecent(train_data,8.35e-10,100000,k)
print(j)
print(l[-1])
plt.plot(i, l, linestyle='-')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Loss')
plt.show()

w1,l,i,j=batchgradientdecent(train_data,8.35e-10,100000,w)
print(j)
print(l[-1])
plt.plot(i, l, linestyle='-')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Loss')
plt.show()


w2,l,i,j=batchgradientdecent(train_data,8.35e-9,1000000,w1)
print(j)
print(l[-1])
plt.plot(i, l, linestyle='-')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Loss')
plt.show()

print(w2) #final weight vector for BGD

y=test_data.iloc[:,-1]
x=test_data.iloc[:,:-1]
x=x.to_numpy()
y=y.to_numpy()
x=np.insert(x, 0, 1, axis=1)
diff=y-np.dot(x,w2)
#print(diff.shape)
diffs=diff**2
loss=np.sum(diffs)/2
print(loss) #test set loss


def stochasticgradientdecent(train_data,r,T,w):
  #vectorized
  y=train_data.iloc[:,-1]
  x=train_data.iloc[:,:-1]
  x=x.to_numpy()
  y=y.to_numpy()
  x=np.insert(x, 0, 1, axis=1)
  w=w#np.zeros(x.shape[1])
  losses=[]
  iter=[]
  for i in range(T):
    random_number = np.random.randint(0, 53)
    xs=x[random_number]
    ys=y[random_number]
    #output.clear()
    iter.append(i)
    diff=ys-np.dot(xs,w)
    #print(diff)
    diffs=diff**2
    loss=np.sum(diffs)/2
    #print(diffs/2,loss)
    losses.append(loss)
    dw=-np.dot(diff.T,xs)
    #t=w
    w=w-r*dw
    #if np.linalg.norm(t-w)<0.0000001:
      #return w,losses,iter,i
    print(loss)
  return w,losses,iter,i


w,l,i,j=stochasticgradientdecent(train_data,8.5e-12,101800,k)
print(j)
print(l[-1])
plt.plot(i, l, linestyle='-')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Loss')
plt.show()


w1,l,i,j=stochasticgradientdecent(train_data,2e-12,101800,w)
print(j)
print(l[-1])
plt.plot(i, l, linestyle='-')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Loss')
plt.show()



w2,l,i,j=stochasticgradientdecent(train_data,9e-12,11000,w1)
print(j)
print(l[-1])
plt.plot(i, l, linestyle='-')
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Loss')
plt.show()


print(w2) # weight SGD


y=test_data.iloc[:,-1]
x=test_data.iloc[:,:-1]
x=x.to_numpy()
y=y.to_numpy()
x=np.insert(x, 0, 1, axis=1)
diff=y-np.dot(x,w2)
#print(diff.shape)
diffs=diff**2
loss=np.sum(diffs)/2
print(loss) #test set loss



y=train_data.iloc[:,-1]
x=train_data.iloc[:,:-1]
x=x.to_numpy()
y=y.to_numpy()
x=np.insert(x, 0, 1, axis=1)
xt=np.transpose(x)
xxt=np.matmul(xt,x)
inverse=np.linalg.inv(xxt)
inverse_x=np.matmul(inverse,xt)
optimal_w=np.matmul(inverse_x,y)
print(optimal_w,'optimal weight')

#Caution lots of print messages