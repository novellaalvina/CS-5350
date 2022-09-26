import math
import pandas as pd
import numpy as np

# features 
label = ["unacc", "acc", "good", "vgood"]
attributes = {'buying':['vhigh', 'high', 'med', 'low'], 'maint':['vhigh', 'high', 'med', 'low'], 'doors': ['2', '3', '4', '5more'], 'persons': ['2', '4', 'more'], 'lug_boot': ['small', 'med', 'big'], 'safety':['low', 'med', 'high']}
columns = ['buying','maint','doors','persons','lug_boot','safety','label']
tree = {}
exampledataset = []

# node 
class Node:
    def __init__(self, name):
        self.name = name
        self.branch = {}

    def isLeaf(self):
        if len(self.branches) == 0:
            return True
        else:
            return False

# reading the data and putting them in dataframe
with open ("train.csv",'r') as f:
  for line in f:
    terms=  line.strip().split(',')

df = pd.read_csv("train.csv", names=['buying','maint','doors','persons','lug_boot','safety','label'])

# Base case \\
# If all examples have same label: return a leaf node with label; 
#                                  if attribute is empty, return a leaf node with most common label
# Otherwise
# create a root node for tree
# find A = attribute in Attributes that best splits S
# for each possible value v of that A can take:
#       Add a new branch corresponding to A = v
#       Let Sv be the subset of example in S with A = v
#           if Sv empty: add leaf node with most common value of Label in S
#           else: below this branch add subtree ID3(Sv, Atttributes - {A}, Label)
#       return root node

#Implement the ID3 algorithm that supports, information gain, majority error and gini index to select attributes for data splits. Besides, your ID3
# should allow users to set the maximum tree depth. Note: you do not need to
# convert categorical attributes into binary ones and your tree can be wide here.

dataset = [[1,2, "good"], [2,3, "bad"], [3,4,"good"], [4,5,"bad"], [3,2,"good"]]
label = ["good","bad"]
probability_set = [0.5,0.75, 0.4]
total_mtd = 12/25
mtd_dict = {0.6:4/9,0.4:0.5}
df8 = pd.DataFrame({
    'x1' : [0,0,0,1,0,1,0], 
    'x2' : [0,1,1,1,0,0,1],
    'x3' : [1,0,1,0,1,0,0],
    'x4' : [0,0,0,0,1,1,1],
    'y': [0,0,0,0,1,1,0]
})

# calculate the p+ and p- of the total dataset
def calc_prob_total(df, lbl_col):
    n = len(df)
    prob_label_total = {}
    for v in df[lbl_col]:
      if v not in prob_label_total:
          prob_label_total[v] = 1
      else:
          prob_label_total[v] += 1
    prob_label_total.update((x, y/n) for x, y in prob_label_total.items())
    return(prob_label_total)  
# calc_prob_total(df8,'y')

# find subset for Attribute = value
def subset(dataset,attribute,attr_value):
    tmp = []
    for row in dataset.index:
        if dataset.iloc[row][attribute] == attr_value:
          tmp.append(row)
    df2 = dataset.iloc[tmp]
    return df2
# subset(df8, 'x2',0)

# calculate probability for A=v
def calc_prob(dataset, value, att, lbl_col,labels):
    n = len(dataset)
    p_y = {}
    val_att = {}
    count = 0

    # to make sure all labels are included
    for label in labels:
      p_y[label] = 0

    # calculate the proportion of positive and negative labels based on the attribute wanted
    for i in range(n):
        # look for the value name wanted
        if dataset.iloc[i][att] == value:
              p_y[dataset.iloc[i][lbl_col]] += 1
              # count += 1
    for p in p_y:
      count += p_y.get(p)
    p_y.update((x, y/count) for x, y in p_y.items())
    val_att[value] = p_y
    return(p_y)
# calc_prob(df8, 0, 'x4','y',labels)
# calc_prob(df8[['x4','y']], 1, 'x4','y')

# calculate majority error
def ME(probability_dict):
    probability_set = []
    for v in probability_dict:
        probability_set.append(probability_dict.get(v))
    return min(probability_set)

# calculate gini index
def GI(probability_dict):
    probability_set = []
    for v in probability_dict:
        probability_set.append(probability_dict.get(v))
    return 1-sum(i*i for i in probability_set)

# calculate information gain
def gain(dataset,attribute,lbl_col,attr_values,labels,func):
    p_total = calc_prob_total(dataset,lbl_col)
    gain_v = func(p_total)
    for val in attr_values:
        Sv = subset(dataset,attribute,val)
        p_val = calc_prob(Sv, val, attribute, lbl_col,labels)
        gain_v -= (len(Sv)/len(dataset)*func(p_val))
    return gain_v

# calculate entropy
def H(probability_dict):
  probability_set = []
  res = 0
  for v in probability_dict:
    probability_set.append(probability_dict.get(v))
  for p in probability_set:
    if p == 0:
      res += 0
    else:
      res += -p*math.log(p,2)
  return res

# calculate best split attribute
def att_bestsplit(dataset,attributes,lbl_col,labels,func):
    attr_gains = {}
    for attribute,attr_values in attributes.items():
        attr_gains[attribute] = gain(dataset,attribute,lbl_col,attr_values,labels,func)
    return max(attr_gains,key=attr_gains.get)
labels = [0,1]
columns = ['x1','x2','x3','x4','y']
attributes = {'x1':[0,1],'x2':[0,1],'x3':[0,1],'x4':[0,1]}
# att_bestsplit(df8,attributes,columns[-1],labels,H)

dataset = df8

# determine the most common value in an attribute
def most_common(dataset,att):
    prob_label_total = {}
    for v in dataset[att]:
      if v not in prob_label_total:
          prob_label_total[v] = 1
      else:
          prob_label_total[v] += 1
    return max(prob_label_total,key=prob_label_total.get)

# ID3
def ID3(dataset, attributes, labels,func,max_depth,current_depth):
    columns = dataset.columns.values.tolist()
    lbl_col = columns[-1]

    # Base case \\
    # If all examples have same label: return a leaf node with label; 
    if(len(labels) == 1):
        leaf_name = str(labels.pop())
        return Node(leaf_name)

    # if attribute is empty, return a leaf node with most common label
    if(len(attributes) == 0):
        return Node(str(most_common(dataset,lbl_col)))
    
    # If reached the max tree depth, return leaf node with most common label.
    if max_depth == current_depth:
        return Node(str(most_common(dataset,lbl_col)))

    # Otherwise
    # find A = attribute in Attributes that best splits S
    A = att_bestsplit(dataset,attributes,lbl_col,labels,func)
    # create a root node for tree
    root = Node(str(A),0,)
    # for each possible value v of that A can take:
    for v in attributes[A]:
        Sv = subset(dataset,A,v)
        
        # if Sv empty: add leaf node with most common value of Label in S
        if len(Sv) == 0:
            root.branches[v] = Node(str(most_common(dataset,lbl_col)))
        # else: below this branch add subtree ID3(Sv, Atttributes - {A}, Label)
        else:
            attributes_v = attributes
            attributes_v.remove(A)

            labels_v = set()
            for data in Sv:
                labels_v.add(data[len(data)-1])

            root.branches[v] = ID3(Sv,attributes_v,labels_v,func,max_depth,current_depth+1)
    return root

# use a DecisionTree to predict the given test example's label. If predicted correctly, return True, otherwise return False
def predict(dataset_line,tree,columns):
    actual_label = dataset_line[len(dataset_line)-1]
    current = tree
    # Go until reach a leaf node
    while not current.isLeaf():
        # Get the attribute to decide on
        decision_attr = current.name 
        # Get the value of that attribute from example
        attr_val = dataset_line[columns.index(decision_attr)]
        # Traverse tree based on example values
        current = current.branches[attr_val] 
    if current.name == actual_label:
        return True
    else:
        return False

