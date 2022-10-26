import copy
import math
import pandas as pd
import numpy as np

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

# tree
class Tree:
    def __init__(self, node):
        self.root = node

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

# find subset for Attribute = value
def subset(dataset,columns, attribute,attr_value, D_t):
    tmp = []
    D_v = []

    # col index for the attribute
    i = columns.index(attribute)

    # seperate the rows with the intended attr val into a subset
    for idx, row in enumerate(dataset):
        if row[i] == attr_value:
            tmp.append(row)
            D_v.append(D_t[idx])

    return tmp, D_v

# calculate probability for A=v
def calc_prob(dataset, labels, D_t):
    n = len(dataset)
    p_y = {}

    # to make sure all labels are included
    for label in labels:
      p_y[label] = 0

    row_idx = 0
    # calculate the proportion of positive and negative labels on the dataset specified
    for row in dataset:
        for l in labels:
            if row[n-1] == l:
                p_y[l] += D_t[row_idx]
        row_idx += 1

    # turn it into fraction
    for x,y in p_y.items():
        if len(dataset)!= 0:
            p_y[x] = y/len(dataset)
    #p_y.update(p_y[x] = y/len(dataset) for x, y in p_y.items() if len(dataset) != 0)
    
    return p_y

# calculate majority error
def ME(probability_dict):
    probability_set = [probability_set.append(probability_dict.get(v)) for v in probability_dict]  
    return min(probability_set)

# calculate gini index
def GI(probability_dict):
    probability_set = [probability_set.append(probability_dict.get(v)) for v in probability_dict]  
    return 1-sum(i*i for i in probability_set)

# calculate information gain
def gain(dataset,columns,attribute,attr_values,labels,D_t,func):
    gain_v = func(dataset,labels, D_t)
    for attr_val in attr_values: # iterate through overcast, rain, sunny for outlook
        s_v = subset(dataset,columns, attribute,attr_val, D_t)
        p_val = calc_prob(s_v,labels,D_t)
        gain_v -= (len(s_v)/len(dataset)*func(p_val))
    return gain_v

# calculate entropy
def H(probability_dict):
    probability_set = [probability_set.append(probability_dict.get(v)) for v in probability_dict]  
    res = 0
    for p in probability_set:
        if p != 0:
          res += -p*math.log(p,2)
    return res

# calculate best split attribute
def att_bestsplit(dataset,columns,attributes,labels,D_t,func):
    attr_gains = {}
    for attribute,attr_values in attributes.items():
        attr_gains[attribute] = gain(dataset,columns,attribute,attr_values,labels,D_t,func)
    return max(attr_gains,key=attr_gains.get)

# determine the most common value in an attribute
def most_common(dataset,attr_idx):
    prob_label_total = {}
    attr_values = set() # can't hv duplicates

    # listing all the attr values 
    for row in dataset:
        attr_val = row[attr_idx]
        attr_values.add(attr_val)

    # standardised all the attr vals in the set
    for attr_val in attr_values:
        prob_label_total[attr_val] = 0

    # counting for the number of attr vals
    for attr_val in attr_values:
        for row in dataset:
          if row[attr_idx] == attr_val:
              prob_label_total[attr_val] += 1

    return max(prob_label_total,key=prob_label_total.get)

# ID3
def ID3(dataset, columns, attributes, labels, D_t, func, max_depth=1,current_depth=0):
    lbl_idx = len(columns)-1

    # Base case 
    # If all examples have same label: return a leaf node with label; 
    if(len(labels) == 1):
        leaf = str(labels.pop())
        return Node(leaf)

    # if attribute is empty, return a leaf node with most common label
    if(len(attributes) == 0):
        return Node(str(most_common(dataset,lbl_idx)))

    # If reached the max tree depth, return leaf node with most common label.
    if max_depth == current_depth:
        return Node(str(most_common(dataset,lbl_idx)))

    # Otherwise
    # find A = attribute in Attributes that best splits S
    A = att_bestsplit(dataset,columns,attributes,labels,D_t,func)
    # create a root node for tree
    root = Node(str(A))
    # for each possible value v of that A can take:
    for attr_val in attributes[A]:

        # make dataset into subset 
        s_v, d_v = subset(dataset,columns, A ,attr_val, D_t)
        
        # if Sv empty: add leaf node with most common value of Label in S
        if len(s_v) == 0:
            root.branches[attr_val] = Node(str(most_common(dataset,lbl_idx)))
        # else: below this branch add subtree ID3(Sv, Atttributes - {A}, Label_subset)
        else:
            attributes_tmp = copy.deepcopy(attributes)
            attributes_tmp.pop(A)

            # to make a new set of labels based on the subset
            labels_v = set()
            for row in s_v:
                labels_v.add(row[len(row)-1])

            root.branches[attr_val] = ID3(s_v, columns, attributes_tmp, labels_v, d_v, func, max_depth,current_depth+1)
    
    return root

# use a DecisionTree to predict the given test example's label, return the label predicted.
def predict(dataset,tree,columns):
    curr = tree
    # Go until reach a leaf node
    while not curr.isLeaf():
        # Get the attribute to decide on
        attr_dec = curr.name 
        # Get the value of that attribute from dataset
        attr_val = dataset[columns.index(attr_dec)]
        # Traverse tree based on example values
        curr = curr.branches[attr_val] 
    return curr.name

# program
def main():
    # # features 
    # labels = ["unacc", "acc", "good", "vgood"]
    # columns = ['buying','maint','doors','persons','lug_boot','safety','label']
    # attributes = {'buying':['vhigh','high','med','low'],
    #                 'maint':['vhigh','high','med','low'],
    #                 'doors':['2','3','4','5more'],
    #                 'persons':['2','4','more'],
    #                 'lug_boot':['small','med','big'],
    #                 'safety':['low','med','high']}
    
    # # storing the dataset
    # dataset = []
    # with open('./car/train.csv', 'r') as train_file:
    #     for line in train_file:
    #         terms = line.strip().split(',')
    #         dataset.append(terms)

    # Labels = set()
    # for example in S:
    #     Labels.add(example[len(example)-1])

    # ID3 
    ### ----------------------------------- ###
    ### NEEDS EDITING FOR ADDING THE WEIGHT ###
    ### ----------------------------------- ###
    root = ID3(S_train,columns,attributes,labels,GI,1,0)
    tree = Tree(root)
    print(tree.root.name)

if __name__ == "__main__":
    main()