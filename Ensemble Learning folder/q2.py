from DT_EL import *
from Adaboost import *

# features 
columns = ['age','job','marital','education','default','balance',
            'housing','loan','contact','day','month','duration',
            'campaign','pdays','previous','poutcome','y']
attributes = {
    'age':['-','+'],
    'job':['admin.','unknown','unemployed','management','housemaid',
        'entrepreneur','student','blue-collar','self-employed','retired',
        'technician','services'],
    'marital':['married','divorced','single'],
    'education':['unknown','secondary','primary','tertiary'],
    'default':['yes','no'],
    'balance':['-','+'],
    'housing':['yes','no'],
    'loan':['yes','no'],
    'contact':['unknown','telephone','cellular'],
    'day':['-','+'],
    'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
    'duration':['-','+'],
    'campaign':['-','+'],
    'pdays':['-','+'],
    'previous':['-','+'],
    'poutcome':['unknown','other','failure','success']
}
label_map = {'no': -1, 'yes': 1, 'negative': 'no', 'positive': 'yes'}
labels = ['yes','no']

# weak_classifiers_end = []
# classifier = None

# program
def adaboost_comparison(columns, attributes,label_map):

    adaboost_sizes = [1,5,10,15,20,50,100,250,500]
    test_errors = []
    training_errors = []

    for size in adaboost_sizes:
        # empty arrays for train and test datasets
        dataset_train = []
        dataset_test = []
        
        # storing the dataset
        dataset_train = []
        with open('./bank/train.csv', 'r') as train_file:
            for line in train_file:
                terms = line.strip().split(',')
                dataset_train.append(terms)
        
        dataset_test = []
        with open('./bank/test.csv', 'r') as test_file:
            for line in test_file:
                terms = line.strip().split(',')
                dataset_test.append(terms)

        # get actual labels
        actual_labels_test = [row[-1] for row in dataset_test]

        # implement adaboost
        classifiers = Adaboost.adaboost(dataset_train, size, columns, attributes,labels)

        # Get the predicted labels
        labels_predicted = [classify_data(classifier, row, label_map) for row in dataset_test]
        test_error = classification_error(labels_predicted, actual_labels_test)
        
        # Calculate the training error
        actual_labels_train = [row[-1] for row in dataset_train]
        labels_predicted = [classify_data(classifiers, row, label_map) for row in dataset_train]
        training_error = classification_error(labels_predicted, actual_labels_train)

        # get classifier error
        test_errors.append(test_error)
        training_errors.append(training_error)
        
    print('Testing Errors: \n')
    for error in test_errors:
        print(error)
    print('\n###################\n')
    print('Training Errors: \n')
    for error in training_errors:
        print(error)

    # adaboost_no_T
    # empty arrays for train and test datasets
    dataset_train = []
    dataset_test = []
    
    # storing the dataset
    dataset_train = []
    with open('./bank/train.csv', 'r') as train_file:
        for line in train_file:
            terms = line.strip().split(',')
            dataset_train.append(terms)
    
    dataset_test = []
    with open('./bank/test.csv', 'r') as test_file:
        for line in test_file:
            terms = line.strip().split(',')
            dataset_test.append(terms)

    test_error_each_iter = []
    actual_labels_test = [row[-1] for row in dataset_test]
    train_error_each_iter = []
    actual_labels_train = [row[-1] for row in dataset_train]
    for weak_classifier in weak_classifiers_end:
        predicted_labels_test = [classify_data(weak_classifier,row) for row in dataset_test]
        predicted_labels_train = [classify_data(weak_classifier,row) for row in dataset_train]  

    # Calculate the classification error
    test_error_each_iter.append(classification_error(predicted_labels_test, actual_labels_test))
    train_error_each_iter.append(classification_error(predicted_labels_train, actual_labels_train))
    

    print('Test error for individual weak classifiers')
    print(test_error_each_iter)

    print('\n###################\n')

    print('Train error for individual weak classifiers')
    print(train_error_each_iter)
        
def classification_error(labels_predicted, actual_labels):
    # Calculate the classification error
    if len(labels_predicted) != len(actual_labels):
        raise ValueError('The number of predicted labels should be equal to the number of correct labels!')

    incorrect_counter = 0
    for i in range(len(labels_predicted)):
        if labels_predicted[i] != actual_labels_test[i]: 
            incorrect_counter += 1
    return incorrect_counter /len(labels_predicted) * 100

def main():
    adaboost_comparison(columns, attributes,label_map)

if __name__ == "__main__":
    main()
