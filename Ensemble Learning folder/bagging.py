from functools import total_ordering
import DT_EL as dt
import statistics
import random

def bagging(dataset, attributes, columns, total_weak_classifiers, total_sample, labels, D_t):
    total_sample = -1
    global weak_classifier
    weak_classifiers = []

    if total_sample == -1:
            total_sample = len(dataset)
    
    # def create_classifier(self, data,  possible_attribute_values, column_headers, num_weak_classifiers, num_samples):
    # Convert data to binary from numerical
    for i in range(len(dataset[0]) - 1):
        if ['-', '+'] == attributes[columns[i]] and dataset[0][i] not in ['-', '+']: # Column is numerical and not adjusted
            # Find the median value
            median_val = statistics.median([int(row[i]) for row in dataset])
            # Replace all numbers with either + or -
            for row in dataset:
                if int(row[i]) >= median_val:
                    row[i] = '+'
                else:
                    row[i] = '-'

    for i in range(total_ordering):
        random_sample = [random.randrange(len(dataset)) for i in range(total_sample)]
        weak_classifier = dt.ID3(dataset, columns, attributes, labels, D_t, dt.GI(dt.calc_prob(dataset,labels)), max_depth=1,current_depth=0)
        weak_classifiers.append(weak_classifier)

    def classify_data(weak_classifiers, row):
        votes = {}
        for weak_classifier in weak_classifiers:
            vote = classify_data(weak_classifier,row)
            if vote not in votes:
                votes[vote] = 0
            votes[vote] += 1

        return max(votes, key=votes.get)
