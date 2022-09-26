# HW1, Section 2, Question 3a.


import HW1q2 as DT
import statistics

S_train = []
S_test = []

Columns = ['age','job','marital','education','default','balance',
            'housing','loan','contact','day','month','duration',
            'campaign','pdays','previous','poutcome','y']

Attributes = {
    'age':['high','low'],
    'job':['admin.','unknown','unemployed','management','housemaid',
        'entrepreneur','student','blue-collar','self-employed','retired',
        'technician','services'],
    'marital':['married','divorced','single'],
    'education':['unknown','secondary','primary','tertiary'],
    'default':['yes','no'],
    'balance':['high','low'],
    'housing':['yes','no'],
    'loan':['yes','no'],
    'contact':['unknown','telephone','cellular'],
    'day':['high','low'],
    'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
    'duration':['high','low'],
    'campaign':['high','low'],
    'pdays':['high','low'],
    'previous':['high','low'],
    'poutcome':['unknown','other','failure','success']
}

Labels = {'yes','no'}

with open('./bank/train.csv', 'r') as train_f:
    for line in train_f:
        terms = line.strip().split(',')
        S_train.append(terms)

df_train = pd.read_csv("./car/train.csv", names=['buying','maint','doors','persons','lug_boot','safety','label'])

with open('./bank/test.csv', 'r') as test_f:
    for line in test_f:
        terms = line.strip().split(',')
        S_test.append(terms)

# Calculate the median values for each of the numeric attributes
medians = {'age':0.0,'balance':0.0,'day':0.0,'duration':0.0,'campaign':0.0,
            'pdays':0.0,'previous':0.0}

for attr in medians.keys():
    S_attr = []
    for example in S_train:
        S_attr.append(float(example[Columns.index(attr)]))
    medians[attr] = statistics.median(S_attr)

# determine the numberic values in the dataset to 'high' or 'low', if they are above or below the median value for that attribute.
for attr,median in medians.items():
    for example in S_train:
        S_attr_val = float(example[Columns.index(attr)])
        if S_attr_val < median:
            example[Columns.index(attr)] = 'low'
        else:
            example[Columns.index(attr)] = 'high'

    for example in S_test:
        S_attr_val = float(example[Columns.index(attr)])
        if S_attr_val < median:
            example[Columns.index(attr)] = 'low'
        else:
            example[Columns.index(attr)] = 'high'


with open('./bank3a_results.csv','w') as res_f:
    res_f.write('Depth,Train_ME,Train_Gini,Train_Entropy,Test_ME,' +
                        'Test_Gini,Test_Entropy\n')


for max_depth in range(1,17):

    me = DT.ID3(df_train, Attributes, Labels,DT.ME,max_depth,0)
    gini = DT.ID3(df_train, Attributes, Labels,DT.GI,max_depth,0)
    entropy = DT.ID3(df_train, Attributes, Labels,DT.H,max_depth,0)

    train_me_success = 0
    train_gini_success = 0
    train_entropy_success = 0
    train_total = 0

    test_me_success = 0
    test_gini_success = 0
    test_entropy_success = 0
    test_total = 0

    for example in S_train:
        if DT.predict(example,me,Columns):
            train_me_success += 1
        if DT.predict(example,gini,Columns):
            train_gini_success += 1
        if DT.predict(example,entropy,Columns):
            train_entropy_success += 1
        train_total += 1

    for example in S_test:
        if DT.predict(example,me,Columns):
            test_me_success += 1
        if DT.predict(example,gini,Columns):
            test_gini_success += 1
        if DT.predict(example,entropy,Columns):
            test_entropy_success += 1
        test_total += 1

    train_me_error = 1-(train_me_success/train_total)
    train_gini_error = 1-(train_gini_success/train_total)
    train_entropy_error = 1-(train_entropy_success/train_total)

    test_me_error = 1-(test_me_success/test_total)
    test_gini_error = 1-(test_gini_success/test_total)
    test_entropy_error = 1-(test_entropy_success/test_total)

    with open('./bank3a_results.csv','a') as res_f:
        res_f.write('{0},{1:0.3f},{2:0.3f},{3:0.3f},{4:0.3f},{5:0.3f},{6:0.3f}\n'.format(
            max_depth,train_me_error,train_gini_error,train_entropy_error,
            test_me_error,test_gini_error,test_entropy_error))

print('Results written to bank3a_results.csv')
