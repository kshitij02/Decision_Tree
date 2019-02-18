#!/usr/bin/env python
# coding: utf-8

# # Part 1

# #### Training decision tree on categorical data.

# In[1]:


import pandas as pd
import numpy as num
import random
import math
import pprint


# In[2]:


def train_test_set(data,precentage):
    indices=data.index.tolist()
    test_size=(int)(precentage*len(data))
    test_set_indices=random.sample(population=indices,k=test_size)
    #print test_size
    #print len(data)
    test_set=data.loc[test_set_indices]
    #print test_set_indices
    train_set=data.drop(test_set_indices)
    return train_set,test_set
list_categorial_attribute=['Work_accident','promotion_last_5years','sales','salary']


# #### Reading and Dividing data in Training and Vaildation data 

# In[3]:


data=pd.read_csv("train.csv")
#data.head()
# train_data ,test_data = train_test_set(data ,0.2)
train_data=data.sample(frac=0.8,random_state=200)
test_data=data.drop(train_data.index)
test_data.head()
# print len(test_data)
# train_data.head()
#test_data.head()


# #### is_pure fuction 
# Tests wheather only one type data is left or not

# In[4]:


def is_pure(data):
    current=data['left']
    if len(set(current))==1:
        return True 
    else: 
        return False
is_pure(train_data)


# #### calc _entropy function 
# Calculates the entropy of given Data  

# In[5]:


def calc_entropy(data):
    current=data['left'].tolist()
    if is_pure(data):
        return 0
    else:

        num_one=current.count(1)
        p=num_one/float(len(data.index))
#         p=(num_one/float(len(current)))
        entropy= ((p*math.log(p,2)) + ((1-p)*math.log(1-p,2)))*-1
#         print entropy
        return entropy
        
calc_entropy(train_data)


# #### calc_information function
# calculates the information in given data with respect each and every atttributes provided

# In[6]:


def calc_information(data,attribute):
    current=set(data[attribute])
    information=0
    for cur in current:
        current_data=data[data[attribute]==cur]
        current_entropy=calc_entropy(current_data)*(len(current_data.index)/float(len(data.index)))
        information=information+current_entropy
    return information
calc_information(train_data,'sales')    


# #### calc_information_gain function
#  Calculates Information Gain of attribute on the given data

# In[7]:


def calc_inforamtion_gain(data,attribute):
    return calc_entropy(data)-calc_information(data,attribute)
calc_inforamtion_gain(train_data,'sales')


# #### best_information_gain function 
# Find best information gain attributes of from all attributes of present in the list_categorial_attribute

# In[8]:


def best_information_gain(data,list_categorial_attribute):
    dict_categorial_attribute={}
    flag=0
    min_v=0
    min_at=''
    for attr in list_categorial_attribute:
        if flag==0:
            min_v=calc_inforamtion_gain(data,attr)
            min_at=attr
            flag=1
        elif min_v<calc_inforamtion_gain(data,attr):
            min_v=min_v=calc_inforamtion_gain(data,attr)
            min_at=attr
    return min_at
best_information_gain(train_data,list_categorial_attribute)


# #### build_tree 
# build_tree fuctions builds tree on the currrently present data in the form of dictionary of dictionary,while recurrsively calling itself again and agian and assigning attribute with best_gain_attribute as root of present tree untill one of following conditions is meet-
# 1. Data size become zero 
# 2. No Attributes is present in attribute_list 
# 3. All data left column value is same
# 

# In[9]:


def build_tree(table, prev_table, attribute_list, tree = None):
    if len(set(table['left'])) <= 1:
        return {'leaf' : table['left'].tolist()[0]}
    elif len(table) == 0:
        return {'leaf': num.unique(prev_table['left'])[num.argmax(num.unique(prev_table['left'], return_counts = True)[1])]}
    elif len(attribute_list) == 0:
        return {'leaf': num.unique(prev_table['left'])[num.argmax(num.unique(prev_table['left'], return_counts = True)[1])]}
    node = best_information_gain(table, attribute_list)
    attribute_list.remove(node)
    if tree is None:
        tree = {}
        tree[node] = {}
    for v in table[node].unique():
        mod_table = table.where(table[node] == v).dropna()
        tree[node][v] = build_tree(mod_table, table, attribute_list[:])
    return tree

tree = build_tree(train_data, train_data, list_categorial_attribute[:])
# pprint.pprint(tree)


# #### predict_helper and predict fucntion :
# Function predict_helper is called on whole validation_data and then predict fucntion is called on each row one by one and  tree build throung train data is used to predict the result of the given row 

# In[10]:


def predict_helper(test_data , li , tree):
#     print(test_data.index)
    for i in test_data.index:
#         print test_data.loc[i]
        predict(test_data.loc[i],li,tree)
    return li
def predict(row,li,tree):
    try:
        if tree.keys()[0]=='leaf':
            li.append(tree['leaf'])
        else :
            t=tree.keys()[0]
    #         print t
            value=row[t]
    #         print tree
    #         print row[t]
    #          print tree[row[t]]
            predict(row,li,tree[t][value])
    except:
        li.append(0.0) #default value
pridected_value=[]
pridected_value= predict_helper(test_data,pridected_value,tree)
# print len(pridected_value)
# print pridected_value.count(1)
# print pridected_value.count(0)


# #### calc_prefomance function
# Calcalutes the preformance of the predict values from the tree produce through the build_tree with respect to it's actual value
# And give Accuracy,Precision ,Recall & F1

# In[11]:


target_value = test_data['left'].tolist()
# print len(target_value)
def calc_preformance(target_value,pridected_value):
    t_p=0
    f_p=0
    t_n=0
    f_n=0
    for i in range(len(target_value)):
        if target_value[i]==0 and target_value[i]==pridected_value[i]:
            t_n=t_n+1
        elif target_value[i]==1 and target_value[i]==pridected_value[i]:
            t_p=t_p+1
        elif pridected_value[i]==1 and target_value[i]==0:
            f_p=f_p+1
        elif pridected_value[i]==0 and target_value[i]==1:
            f_n=f_n+1
    accuracy=(t_n+t_p)/float(t_n+t_p+f_p+f_n)
    
    precison=(t_p)/float(t_p+f_p)
    recall=(t_p)/float(t_p+f_n)
    a=1/precison
    b=1/recall
    f1_score=2/(a+b)
    print "Accuracy ",accuracy
    print "Precision ",precison
    print "Recall ",recall
    print "F1 Score",f1_score
    


# ### Accuracy Precision Recall F1 Score

# In[12]:


calc_preformance(target_value,pridected_value)


# In[ ]:




