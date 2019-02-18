#!/usr/bin/env python
# coding: utf-8

# # Part 3
# #### Contrasting the effectiveness of Misclassification rate, Gini, Entropy 

# In[89]:


import pandas as pd
import numpy as num
import random
import math
import pprint


# In[90]:


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
list_numeric_attribute=['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company'
]
dic_avg_numeric={}


# #### Reading and Dividing data in Training and Vaildation data 

# In[91]:


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

# In[92]:


def is_pure(data):
    current=data['left']
    if len(set(current))==1:
        return True 
    else: 
        return False
is_pure(train_data)


# #### calc _entropy function 
# Calculates the entropy of given Data when type calc == 0 , gini of given Data when type calc == 1 & missclassification of given Data when type calc == 2

# In[93]:


def calc_entropy(data,type_calc):
    current=data['left'].tolist()
    if is_pure(data):
        return 0
    else:
        num_one=current.count(1)
        p=num_one/float(len(data.index))
        if type_calc==0:
            entropy= ((p*math.log(p,2)) + ((1-p)*math.log(1-p,2)))*-1
            return entropy
        elif type_calc==1:
            gini_index=2*p*(1-p)
            return gini_index
        elif type_calc==2:
            missclasification=min(p,1-p)
            return missclasification
calc_entropy(train_data,1)


# #### calc_information function
# calculates the information in given data with respect each and every atttributes provided

# In[94]:


def calc_information(data,attribute,type_calc):
    current=set(data[attribute])
    information=0
    for cur in current:
        current_data=data[data[attribute]==cur]
        current_entropy=calc_entropy(current_data,type_calc)*(len(current_data.index)/float(len(data.index)))
        information=information+current_entropy
    return information
calc_information(train_data,'sales',1)    


# #### calc_information_gain function
#  Calculates Information Gain of attribute on the given data

# In[95]:


def calc_inforamtion_gain(data,attribute,type_calc):
    return calc_entropy(data,type_calc)-calc_information(data,attribute,type_calc)
calc_inforamtion_gain(train_data,'sales',1)


# #### best_information_gain function 
# Find best information gain attributes of from all attributes of present in the list_categorial_attribute

# In[96]:


def best_information_gain(data,list_categorial_attribute,type_calc):
    dict_categorial_attribute={}
    flag=0
    min_v=0
    min_at=''
    for attr in list_categorial_attribute:
        if flag==0:
            min_v=calc_inforamtion_gain(data,attr,type_calc)
            min_at=attr
            flag=1
        elif min_v<calc_inforamtion_gain(data,attr,type_calc):
            min_v=min_v=calc_inforamtion_gain(data,attr,type_calc)
            min_at=attr
    return min_at
best_information_gain(train_data,list_categorial_attribute,1)


# #### covert_numeric_to_categorical function
# Fucntion converts the numeric data to categorical data by assigning labels to the in place of numeric values of the attributes.
# First data is sorted according to the numeric value attribute and the all unique numeric values present in the numerical value attribute is obtained then for all the unique value we find out what is maximum number of time occuring in 'left' attribute of data and that value is assign to that unique value.And then data is accessed is sequential and where the value assigned to numeric value changes the average of numerical value is taken and value less that value is given one label and value greater than that avgerage falls under other label
# 

# In[97]:


def convert_numeric_to_categorical(data,attribute):
    data.sort_values(by=[attribute],inplace=True)
#     print data
    dic={}
    li=data[attribute].unique().tolist()
    for l in li :
        temp=data[data[attribute]==l]
        dic[l]=num.unique(temp['left'])[num.argmax(num.unique(temp['left'], return_counts = True)[1])]
#     print dic                                               
    flag=0
    avg_list=[]
    count=0
    for i in data.index :
        if flag==0:
            prev_elem=data.loc[i,attribute]
            prev_left=dic[prev_elem]
            flag=1
        elif prev_left!=dic[data.loc[i,attribute]]:
            avg_list.append((prev_elem+data.loc[i,attribute])/2.0)
            prev_elem=data.loc[i,attribute]
            prev_left=dic[prev_elem]
            count=count+1
#             print count
        else :
            prev_elem=data.loc[i,attribute]
            prev_left=dic[prev_elem]
#     print avg_list
    dic_avg_numeric[attribute]=avg_list
    label=0
    list_label=[]
    index_list=[]
    f=0
    for i in data.index :
        label=0
        index_list.append(i)
        f=0
        for j in avg_list:
            if data.loc[i,attribute]<=j:
                list_label.append(label)
                f=1
                break
            else :
                label=label+1
        if f==0:
            list_label.append(label)
#     print pd.Series(list_label)
    
#     print len(data)
#     print data.head()
    
    num_attribute=attribute+'_numeric'
    data[num_attribute]=pd.Series(list_label,index=index_list)
    
#     print data.head()
def label_prediction(data,attribute):
    avg_list=dic_avg_numeric[attribute]
    label=0
    list_label=[]
    index_list=[]
    f=0
    for i in data.index :
        label=0
        index_list.append(i)
        f=0
        for j in avg_list:
            if data.loc[i,attribute]<=j:
                list_label.append(label)
                f=1
                break
            else :
                label=label+1
        if f==0:
            list_label.append(label)
    num_attribute=attribute+'_numeric'
    data[num_attribute]=pd.Series(list_label,index=index_list)



for attr in list_numeric_attribute:
     convert_numeric_to_categorical(train_data,attr)
# convert_numeric_to_categorical(train_data,'satisfaction_level')
# print train_data


# #### build_tree 
# build_tree fuctions builds tree on the currrently present data in the form of dictionary of dictionary,while recurrsively calling itself again and agian and assigning attribute with best_gain_attribute as root of present tree untill one of following conditions is meet-
# 1. Data size become zero 
# 2. No Attributes is present in attribute_list 
# 3. All data left column value is same
# 

# In[98]:


def build_tree(table, prev_table, attribute_list,type_calc,tree = None):
    if len(set(table['left'])) <= 1:
        return {'leaf' : table['left'].tolist()[0]}
    elif len(table) == 0:
        return {'leaf': num.unique(prev_table['left'])[num.argmax(num.unique(prev_table['left'], return_counts = True)[1])]}
    elif len(attribute_list) == 0:
        return {'leaf': num.unique(prev_table['left'])[num.argmax(num.unique(prev_table['left'], return_counts = True)[1])]}
    node = best_information_gain(table, attribute_list,type_calc)
    attribute_list.remove(node)
    if tree is None:
        tree = {}
        tree[node] = {}
    for v in table[node].unique():
        mod_table = table.where(table[node] == v).dropna()
        tree[node][v] = build_tree(mod_table, table, attribute_list[:],type_calc)
    return tree

list_final=list_categorial_attribute
for i in list_numeric_attribute:
    list_final.append(i+'_numeric')
    
tree_entropy = build_tree(train_data, train_data, list_final[:],0)
tree_gini = build_tree(train_data, train_data, list_final[:],1)
tree_missclassification = build_tree(train_data, train_data, list_final[:],2)
# pprint.pprint(tree_entropy)


# #### predict_helper and predict fucntion :
# Function predict_helper is called on whole validation_data and then predict fucntion is called on each row one by one and  tree build throung train data is used to predict the result of the given row 

# In[99]:


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
# print len(pridected_value)
# print pridected_value.count(1)
# print pridected_value.count(0)
for attr in list_numeric_attribute:
     label_prediction(test_data,attr)
# convert_numeric_to_categorical(test_data,'satisfaction_level')
pridected_value_entropy=[]
pridected_value_entropy= predict_helper(test_data,pridected_value_entropy,tree_entropy)
pridected_value_gini=[]
pridected_value_gini= predict_helper(test_data,pridected_value_gini,tree_gini)
pridected_value_missclassification=[]
pridected_value_missclassification= predict_helper(test_data,pridected_value_missclassification,tree_missclassification)


# #### calc_prefomance function
# Calcalutes the preformance of the predict values from the tree produce through the build_tree with respect to it's actual value
# And give Accuracy,Precision ,Recall & F1

# In[100]:


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
#     print "ture positive",t_p
#     print "false positive",f_p
#     print "false negative",f_n
#     print "ture negative",t_n
    
    print "Accuracy ",accuracy
    print "Precision ",precison
    print "Recall ",recall
    print "F1 Score",f1_score


# ### Accuracy Precision Recall F1 Score with Entropy

# print ("Entropy")
# calc_preformance(target_value,pridected_value_entropy)
# 

# ### Accuracy Precision Recall F1 Score with Gini Index

# In[102]:


print ("Gini Index")
calc_preformance(target_value,pridected_value_gini)


# ### Accuracy Precision Recall F1 Score with Missclassification

# In[103]:


print ("Missclassification Rate")
calc_preformance(target_value,pridected_value_missclassification)

