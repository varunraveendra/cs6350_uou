from Decisiontree import *


def q1():
  terms=[]
  with open('car-4/data-desc.txt','r') as f:
    for line in f:
      terms.append(line)
    terms.remove('\n')
    g=terms.index('| columns\n')
    l=terms.index('| label values\n')
    label=terms[l+1].strip().split(', ')
    header=terms[g+1].strip().split(',')

  train_data=pd.read_csv('car-4/train.csv',names=header)
  test_data=pd.read_csv('car-4/test.csv',names=header)

  return generatereport(train_data,test_data,6)

def q2a():
  
 header=['age','job','martial','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','label']
 train_data=pd.read_csv('/bank-4/train.csv',names=header)
 test_data=pd.read_csv('/bank-4/test.csv',names=header)
 preprocess_bankdata(train_data)
 preprocess_bankdata(test_data)
 generatereport(train_data,test_data,16)

def q2b():
 header=['age','job','martial','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','label']
 train_data=pd.read_csv('/bank-4/train.csv',names=header)
 test_data=pd.read_csv('/bank-4/test.csv',names=header)
 preprocess_bankdata(train_data)
 preprocess_bankdata(test_data)
 replace_unknown(train_data)
 replace_unknown(train_data)

 generatereport(train_data,test_data,16)


import sys

n = int(sys.argv[1])
if n ==1:
 q1()
elif n==2:
 q2a()
elif n==3:
 q2b()




