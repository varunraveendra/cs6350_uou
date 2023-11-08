import numpy as np
import pandas as pd
header=['age','job','martial','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','label']
train_data=pd.read_csv('bank-4/train.csv',names=header)
test_data=pd.read_csv('bank-4/test.csv',names=header)

def preprocess_bankdata(train_data):
 numeric_cols = train_data.select_dtypes(include=['int', 'float']).columns.tolist()
 for i in numeric_cols:
  train_data[i].mask(train_data[i] <= train_data.median(numeric_only=True)[numeric_cols.index(i)],1, inplace=True)
  train_data[i].mask(train_data[i] !=1,0, inplace=True)

def preprocess_for_boosting(train_data):
  train_data['label'].mask(train_data['label'] =='yes',1, inplace=True)
  train_data['label'].mask(train_data['label'] !=1,-1, inplace=True)
  train_data["label"] = train_data["label"].astype(str).astype(int)
  train_data['Weights']=1/len(train_data)

def total_entropy(train_data,heuristics):#####
  values=train_data['label'].unique()
  datapoints=train_data["Weights"].sum()
  entropy=0
  me=[]
  for i in values:
    lablecount=train_data[train_data['label']==i]["Weights"].sum()
    if heuristics==0:
     ventropy=-(lablecount/datapoints)*np.log2(lablecount/datapoints)
     entropy+=ventropy
    elif heuristics==1:
      ventropy=-((lablecount/datapoints)**2)
      entropy+=ventropy
    elif heuristics==2:
      me.append((lablecount/datapoints))
    #print(ventropy)
  if heuristics==0:
   return entropy
  if heuristics==1:
    return (1+entropy)
  if heuristics==2:
    return (1-max(me))

def maxinfogain(train_data,te,heuristics):#########
 infogain=[]
 label=train_data['label'].unique()
 for i in train_data.columns[:-2]:
   fval=train_data[i].unique()
   tot=train_data["Weights"].sum()
   att_ent=0
   for j in fval:
     att_val_count=train_data[train_data[i]==j]["Weights"].sum()
     att_val=train_data[train_data[i]==j]
     tentropy=0
     me1=[]

     for m in att_val['label'].unique():
       j_count=att_val[att_val['label']==m]["Weights"].sum()
       if j_count==0:
        print("herere",j_count)
       if j_count!=0:
        if heuristics==0:
          #print('works')
          ventropy=-(j_count/att_val_count)*np.log2(j_count/att_val_count)
          tentropy+=ventropy
        elif heuristics==1:
          ventropy=-((j_count/att_val_count)**2)
          tentropy+=ventropy
        elif heuristics==2:
          me1.append((j_count/att_val_count))
        #print(j_count,att_val_count)
        #tentropy+=ventropy
        #print(tentropy,ventropy)
     if heuristics==0:
      att_val_ent=(att_val_count/tot)*tentropy
     if heuristics==1:
      att_val_ent=(att_val_count/tot)*(1+tentropy)
     if heuristics==2:
      att_val_ent=(att_val_count/tot)*(1-max(me1))
     #print(att_val_ent,att_val_count,tot,'this')
     att_ent+=att_val_ent
     #print(i,j,att_ent,att_val_ent)
   #print(att_ent,"outloop")
   infogain.append(te-att_ent)
  #  if (te-att_ent)==te:
  #   return i
 return train_data.columns[:-2][np.argsort(infogain)[-1]]

def buildTree(rf,train_data,depth=7,h=0,lim=0,tree=None):
    n=lim
    Class = train_data.keys()[-1]
    node = maxinfogain(train_data,total_entropy(train_data,h),h,rf)
    #print(node)
    n=n+1
    attValue = np.unique(train_data[node])
    if tree is None:
        tree={}
        tree[node] = {}
    for value in attValue:
        subtable =train_data[train_data[node] == value].reset_index(drop=True)
        subtable=subtable.drop([node],axis=1)
        #print(subtable)
        clValue,counts = np.unique(subtable['label'],return_counts=True)
        #print(clValue,counts,"checjkjdalijolkaw")
        if len(counts)==1:
            tree[node][value] = clValue[0]
        elif len(subtable.columns[:-2])==0:
          tree[node][value] = clValue[np.random.randint(0, 2)]
        elif n<depth:
            #print('called',node,value,n)
            tree[node][value] = buildTree(rf,subtable,depth,h,n)
        elif n==depth:
            #print('cut',node,value,n)
            tree[node][value]= np.sign((subtable['label']*subtable['Weights']).sum())
            #print(np.sign((subtable['label']*subtable['Weights']).sum())).   subtable.loc[subtable['Weights'].idxmax(),['label']].values
    return tree

def predict(tree, instance):
  
    if not isinstance(tree, dict):
        return tree
    else:
        root_node = next(iter(tree))
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance)
        else:
            def get_leaf_node_values(tree):
              values = []
              for key, value in tree.items():
                if isinstance(value, dict):
                  values += get_leaf_node_values(value)
                else:
                  values.append(value)
              return values
            v=get_leaf_node_values(tree)
            return max(set(v), key = v.count)


def evaluate(tree, test_data_m):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.loc[index])
        if result == test_data_m['label'].iloc[index]:
            correct_preditct += 1
        else:
            wrong_preditct += 1
    err = wrong_preditct / (correct_preditct + wrong_preditct)
    return err


def evaluate_adaboost(tree,alp,test_data_m,currentada):
  correct_preditct = 0
  wrong_preditct = 0
  for index, row in test_data_m.iterrows():
    #for t,a in zip(alltrees,alps):
    result=predict(tree,test_data_m.loc[index])
    #print(currentada[index],"init")
    currentada[index]+=alp*result
    #print(currentada[index],"final")
    #ress+=(result*a)
    f_result = np.sign(currentada[index])
    if f_result == test_data_m['label'].iloc[index]:
            correct_preditct += 1
    else:
            wrong_preditct += 1

  error = wrong_preditct / (correct_preditct + wrong_preditct)
  return error, currentada


def evaluate_and_update_weights(tree, test_data_m):
    correct_preditct = 0
    wrong_preditct = 0
    mindex=[]
    #windex=[]
    werror=0
    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.loc[index])
        if result == test_data_m['label'].iloc[index]:
            correct_preditct += 1
            mindex.append(1)
        else:
            wrong_preditct += 1
            werror+=test_data_m.loc[index,['Weights']].values[0]
            mindex.append(-1)
    accuracy = correct_preditct / (correct_preditct + wrong_preditct)
    error= wrong_preditct/ (correct_preditct + wrong_preditct)
    #print(werror)
    alphat=(1/2)*(np.log((1-werror)/werror))
    #print(len(cindex),len(windex))
    #print(mindex,alphat)
    #print(np.exp(np.multiply(alphat,mindex)))
    test_data_m['Weights']*=np.exp(np.multiply(-alphat,mindex))
    test_data_m['Weights']/=test_data_m['Weights'].sum()
    #check=test_data_m['Weights'].sum()
    #print(check,"is it 1???")
    return alphat,error


preprocess_bankdata(train_data)
preprocess_bankdata(test_data)

preprocess_for_boosting(train_data)
preprocess_for_boosting(test_data)

import matplotlib.pyplot as plt

def baggingdecisiontree(train_data,test_data,subdatasz,rep,T):
  resultstr=[]
  resultste=[]
  iter=[]
  trees=[]
  currentadatr=np.zeros(len(train_data))
  currentadate=np.zeros(len(test_data))
  for i in range(1,T+1):
    iter.append(i)
    subset = train_data.sample(n=subdatasz, replace=rep)
    t=buildTree(0,subset,50000,0)
    trees.append(t)
    #print(i)
  # return resultstr
    adatr,currentadatr=evaluate_adaboost(t,1,train_data,currentadatr)
    adate,currentadate=evaluate_adaboost(t,1,test_data,currentadate)
    resultstr.append(adatr)
    resultste.append(adate)
    print(i,adatr,adate)
  plt.plot(iter, resultstr, label='Train Error', linestyle='-')
  plt.plot(iter, resultste, label='Test Error', linestyle='-')
  plt.xlabel('Number of Iterations')
  plt.ylabel('Error')
  plt.title('Bagging')
  plt.legend()
  plt.show()


def rfdecisiontree(train_data,test_data,subdatasz,rep,T,rf):
  resultstr=[]
  resultste=[]
  iter=[]
  trees=[]
  currentadatr=np.zeros(len(train_data))
  currentadate=np.zeros(len(test_data))
  for i in range(1,T+1):
    iter.append(i)
    subset = train_data.sample(n=subdatasz, replace=rep)
    t=buildTree(rf,subset,50000,0)
    trees.append(t)
    #print(i)
  # return resultstr
    adatr,currentadatr=evaluate_adaboost(t,1,train_data,currentadatr)
    adate,currentadate=evaluate_adaboost(t,1,test_data,currentadate)
    if i ==1:
      stree=currentadate
    resultstr.append(adatr)
    resultste.append(adate)
    print(i,adatr,adate)
  plt.plot(iter, resultstr, label='Train Error', linestyle='-')
  plt.plot(iter, resultste, label='Test Error', linestyle='-')
  plt.xlabel('Number of Iterations')
  plt.ylabel('Error')
  plt.title('Forest')
  plt.legend()
  plt.show()
  return stree, currentadate


def adaboostdecisiontree(train_data,test_data,T):
  sterror=[]
  resultstr=[]
  resultste=[]
  st_test_err=[]
  iter=[]
  currentadatr=np.zeros(len(train_data))
  currentadate=np.zeros(len(test_data))
  for i in range(1,T+1):
    iter.append(i)
    t=buildTree(0,train_data,2,0)
    alpt,trerror=evaluate_and_update_weights(t,train_data)
    testerr=evaluate(t,test_data)
    sterror.append(trerror)
    st_test_err.append(testerr)
    adatr,currentadatr=evaluate_adaboost(t,alpt,train_data,currentadatr)
    adate,currentadate=evaluate_adaboost(t,alpt,test_data,currentadate)
    resultstr.append(adatr)
    resultste.append(adate)
    print(i,adatr,adate)
  plt.subplot(1, 2, 1)
  plt.plot(iter, resultstr, label='Train Error', linestyle='-')
  plt.plot(iter, resultste, label='Test Error', linestyle='-')
  plt.xlabel('Number of Iterations')
  plt.ylabel('Error')
  plt.title('ADABOOST')
  plt.subplot(1, 2, 2)
  plt.plot(iter, sterror, label='Train Error', linestyle='-')
  plt.plot(iter, st_test_err, label='Test Error', linestyle='-')
  plt.xlabel('Number of Iterations')
  plt.ylabel('Error')
  plt.title('DECISION TREE INDIVIDUAL')
  plt.legend()
  plt.show()

stree6,rf6=rfdecisiontree(train_data,test_data,5000,True,500,6) #RF 6
stree4,rf4=rfdecisiontree(train_data,test_data,5000,True,500,4) #RF 4
stree2,rf2=rfdecisiontree(train_data,test_data,5000,True,500,2) #RF 2

print(rf2,rf4,rf6)

resultbag=[rf2,rf4,rf6]


biases=[]
variances=[]


for index, row in test_data.iterrows():
 restot=0
 reslist=[]
 for i in range(0,3):
  res=resultbag[i][index] 
  if res>0:
    res=1
  else:
    res=-1
  restot+=res
  reslist.append(res)
 avg=restot/3
 if avg>0:
   avg=1
 else:
   avg=-1
 sqerror=[((x - avg) ** 2) for x in reslist]
 var=sum(sqerror)/3
 biases.append(test_data['label'].iloc[index]-avg)
 variances.append(var)
 #print(index)

print('total bias=',sum(biases)/len(biases))
print('total variance=',sum(variances)/len(variances))
print("GSE=",sum(biases)/len(biases)+sum(variances)/len(variances))

stree6,rf6=rfdecisiontree(train_data,test_data,5000,True,1,6) #RF 6
stree4,rf4=rfdecisiontree(train_data,test_data,5000,True,1,4) #RF 6
stree2,rf2=rfdecisiontree(train_data,test_data,5000,True,1,2) #RF 6
resultbag=[rf2,rf4,rf6]


biases=[]
variances=[]


for index, row in test_data.iterrows():
 restot=0
 reslist=[]
 for i in range(0,3):
  res=resultbag[i][index] 
  if res>0:
    res=1
  else:
    res=-1
  restot+=res
  reslist.append(res)
 avg=restot/3
 if avg>0:
   avg=1
 else:
   avg=-1
 sqerror=[((x - avg) ** 2) for x in reslist]
 var=sum(sqerror)/3
 biases.append(test_data['label'].iloc[index]-avg)
 variances.append(var)
 #print(index)

print('total bias=',sum(biases)/len(biases))
print('total variance=',sum(variances)/len(variances))
print("GSE=",sum(biases)/len(biases)+sum(variances)/len(variances))

####   Code takes several hours to run