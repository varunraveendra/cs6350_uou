import numpy as np
import pandas as pd

def preprocess_bankdata(train_data):
 numeric_cols = train_data.select_dtypes(include=['int', 'float']).columns.tolist()
 for i in numeric_cols:
  train_data[i].mask(train_data[i] <= train_data.median(numeric_only=True)[numeric_cols.index(i)],1, inplace=True)
  train_data[i].mask(train_data[i] !=1,0, inplace=True)


def replace_unknown(train_data):
 for i in train_data.columns[:-1]:
   vals=train_data[i].unique()
   count=[]
   if isinstance(vals[0], str):
     #print(i)
     wunk=[]
     for j in vals:
       #print(j)
       if j!='unknown':
         wunk.append(j)
         #print(j)
         count.append(train_data[train_data[i]==j].shape[0])
     train_data[i].mask(train_data[i] == 'unknown', wunk[np.argsort(count)[-1]], inplace=True)    

def total_entropy(train_data,heuristics):#####
  values=train_data['label'].unique()
  datapoints=train_data.shape[0]
  entropy=0
  me=[]
  for i in values:
    lablecount=train_data[train_data['label']==i].shape[0]
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
 for i in train_data.columns[:-1]:
   fval=train_data[i].unique()
   tot=train_data.shape[0]
   att_ent=0
   for j in fval:
     att_val_count=train_data[train_data[i]==j].shape[0]
     att_val=train_data[train_data[i]==j]
     tentropy=0
     me1=[]
     for m in label:
       j_count=att_val[att_val['label']==m].shape[0]
       if j_count!=0:
        if heuristics==0:
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
 return train_data.columns[:-1][np.argsort(infogain)[-1]]

def buildTree(train_data,depth=7,h=0,lim=0,tree=None):
    n=lim
    Class = train_data.keys()[-1]
    node = maxinfogain(train_data,total_entropy(train_data,h),h)
    n=n+1
    attValue = np.unique(train_data[node])    
    if tree is None:                    
        tree={}
        tree[node] = {}
    for value in attValue:    
        subtable =train_data[train_data[node] == value].reset_index(drop=True)
        clValue,counts = np.unique(subtable['label'],return_counts=True)                              
        if len(counts)==1:
            tree[node][value] = clValue[0]                                                    
        elif n<depth:          
            #print('called',node,value,n)               
            tree[node][value] = buildTree(subtable,depth,h,n)
        elif n==depth:
            #print('cut',node,value,n) 
            tree[node][value]= clValue[np.argsort(counts)[-1]]                
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
            return None
        
def evaluate(tree, test_data_m):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows(): 
        result = predict(tree, test_data_m.iloc[index]) 
        if result == test_data_m['label'].iloc[index]: 
            correct_preditct += 1 
        else:
            wrong_preditct += 1
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) 
    return accuracy



from tabulate import tabulate
def generatereport(train_data,test_data,maxtreesize):
 print("ENTROPY:        Building and Evaluating.......")
 mydata = []
 n=0
 while n<maxtreesize:
  t=buildTree(train_data,n+1,0)
  tracc=evaluate(t,train_data)
  teacc=evaluate(t,test_data)
  lis=[n+1,tracc,teacc]
  mydata.append(lis)
  n+=1
 head = ["Tree Depth", "Train Accuracy","Test Accuracy"]
 print(tabulate(mydata, headers=head, tablefmt="grid"))
 #print('---------------------------------------------------------')
 print('GINI INDEX:    Building and Evaluating.......')
 mydata=[]
 n=0
 while n<maxtreesize:
  t=buildTree(train_data,n+1,1)
  tracc=evaluate(t,train_data)
  teacc=evaluate(t,test_data)
  lis=[n+1,tracc,teacc]
  mydata.append(lis)
  n+=1
 head = ["Tree Depth", "Train Accuracy","Test Accuracy"]
 print(tabulate(mydata, headers=head, tablefmt="grid"))
 #print('---------------------------------------------------------')
 print('MAJORITY ERROR:    Building and Evaluating.......')
 mydata=[]
 n=0
 while n<maxtreesize:
  t=buildTree(train_data,n+1,2)
  tracc=evaluate(t,train_data)
  teacc=evaluate(t,test_data)
  lis=[n+1,tracc,teacc]
  mydata.append(lis)
  n+=1
 head = ["Tree Depth", "Train Accuracy","Test Accuracy"]
 print(tabulate(mydata, headers=head, tablefmt="grid"))


#======================================================================
# run code by changing values here:


#-For Q1 --------------(uncomment)----------------------------------------------

# terms=[]
# with open('/content/drive/MyDrive/car-4/data-desc.txt','r') as f:
#   for line in f:
#     terms.append(line)
#   terms.remove('\n')
#   g=terms.index('| columns\n')
#   l=terms.index('| label values\n')
#   label=terms[l+1].strip().split(', ')
#   header=terms[g+1].strip().split(',')

# train_data=pd.read_csv('/content/drive/MyDrive/bank-4/train.csv',names=header)
# test_data=pd.read_csv('/content/drive/MyDrive/bank-4/test.csv',names=header)

#generatereport(train_data,test_data,6)

#-----------------------------------------------------------------------------------------


#-For Q2 a)-------(uncomment)---------------------------------------------------

#header=['age','job','martial','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','label']
#train_data=pd.read_csv('/content/drive/MyDrive/bank-4/train.csv',names=header)
#test_data=pd.read_csv('/content/drive/MyDrive/bank-4/test.csv',names=header)
#preprocess_bankdata(train_data)
#preprocess_bankdata(test_data)

#-For Q2 b)--------------(uncomment both a) and b) for b)--------------------------------------------

#replace_unknown(train_data)
#replace_unknown(train_data)

#----------------------------------------------------------------------------------------

#generatereport(train_data,test_data,16)

#------------------------------------------------------------------------------------------------
#EOF