# HW1, Section 2, Question 2
# Novella Alvina

import HW1q2 as DT
import math
import pandas as pd

columns = ['buying','maint','doors','persons','lug_boot','safety','label']

attributes = {'buying':['vhigh','high','med','low'],
                'maint':['vhigh','high','med','low'],
                'doors':['2','3','4','5more'],
                'persons':['2','4','more'],
                'lug_boot':['small','med','big'],
                'safety':['low','med','high']}

# with open('train.csv', 'r') as train_f:
#     for line in train_f:
#         terms = line.strip().split(',')
#         S.append(terms)

dataset = pd.read_csv("./car/train.csv", names=['buying','maint','doors','persons','lug_boot','safety','label'])

labels = set()
for data in dataset:
    labels.add(data[len(data)-1])

with open('./car/car_results.csv','w') as res_f:
    res_f.write('Depth,Train_ME,Train_Gini,Train_Entropy,Test_ME,' +
                        'Test_Gini,Test_Entropy\n')

for max_depth in range(1,7):
    
    me = DT.ID3(dataset, attributes, labels,DT.ME,max_depth,0)
    gini = DT.ID3(dataset, attributes, labels,DT.GI,max_depth,0)
    entropy = DT.ID3(dataset, attributes, labels,DT.H,max_depth,0)

    train_me_success = 0
    train_gini_success = 0
    train_entropy_success = 0
    train_total = 0

    test_me_success = 0
    test_gini_success = 0
    test_entropy_success = 0
    test_total = 0

    with open('./car/train.csv','r') as test_f:
        for line in test_f:
            example = line.strip().split(',')
            if DT.predict(example,me,columns):
                train_me_success += 1
            if DT.predict(example,gini,columns):
                train_gini_success += 1
            if DT.predict(example,entropy,columns):
                train_entropy_success += 1
            train_total += 1

    with open('./car/test.csv','r') as test_f:
        for line in test_f:
            example = line.strip().split(',')
            if DT.predict(example,me,columns):
                test_me_success += 1
            if DT.predict(example,gini,columns):
                test_gini_success += 1
            if DT.predict(example,entropy,columns):
                test_entropy_success += 1
            test_total += 1

    train_me_error = 1-(train_me_success/train_total)
    train_gini_erorr = 1-(train_gini_success/train_total)
    train_entropy_error = 1-(train_entropy_success/train_total)

    test_me_error = 1-(test_me_success/test_total)
    test_gini_error = 1-(test_gini_success/test_total)
    test_entropy_error = 1-(test_entropy_success/test_total)

    with open('./car/car_results.csv','a') as res_f:
        res_f.write('{0},{1:0.3f},{2:0.3f},{3:0.3f},{4:0.3f},{5:0.3f},{6:0.3f}\n'.format(
            max_depth,train_me_error,train_gini_error,train_entropy_error,
            test_me_error,test_gini_error,test_entropy_error))

print('Results written to car_results.csv')

