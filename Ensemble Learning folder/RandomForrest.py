import DT-EL as dt
from random import randrange
from statistics import median

# def random_sample_attributes(columns, G):
#     random_attributes = [x for x in range(len(column_headers) - 1)]
#     for i in range(len(columns) - G - 1):
#         del random_attributes[randrange(0, len(random_attributes))]
#     return random_attributes

def RandomForrest(data, attributes, columns, num_weak_classifiers, G,labels,D_t):
    global weak_classifiers 
    weak_classifiers = []

    # Convert data to binary from numerical
    for i in range(len(data[0]) - 1):
        if ['-', '+'] == attributes[columns[i]] and dataset[0][i] not in ['-', '+']: # Column is numerical and not adjusted
            # Find the median value
            median_val = median([int(row[i]) for row in data])
            # Replace all numbers with either + or -
            for row in data:
                if int(row[i]) >= median_val:
                    row[i] = '+'
                else:
                    row[i] = '-'
    for i in range(num_weak_classifiers):
        random_sample = [randrange(len(dataset)) for i in range(total_sample)]
        random_attributes = [x for x in range(len(columns) - 1)]
        random_attributes = [random_attributes[randrange(0, len(random_attributes))] for i in range(len(columns - G - 1))]
        weak_classifier = dt.ID3(random_sample, columns,random_attributes=random_attributes, labels, D_t, dt.GI(dt.calc_prob(dataset,labels)), max_depth=1,current_depth=0)
        weak_classifiers.append(weak_classifier)
    
    def classify_data(classifier, row):
        votes = {}
        for weak_classifier in weak_classifiers:
            vote = classify_data(weak_classifier,row)
            if vote not in votes:
                votes[vote] = 0
            votes[vote] += 1

        return max(votes, key=votes.get)
