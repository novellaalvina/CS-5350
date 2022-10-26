import numpy as np
import random
import matplotlib.pyplot as plt



# compute gradient, J(w)
def grad_Jw(y_i, w, x_i,x_ij):
    return (y_i-(np.dot(w,x_i)))*x_ij

# update w, i.o.w, compute w^(t+1)
def update_weight(w, r, j_w):
    return w - r*j_w

# compute cost function
def cost_func(w_T, dataset):
    error = 0
    for row in dataset:
        error += 0.5 * (row[-1] - np.dot(w_T, row[:len(row)-1])) ** 2
    return error

# take dataset with no y
def no_y_dataset(dataset):
  dataset_no_y = []
  for row in dataset:
    dataset_no_y.append(row[0:len(row)-1])
  return dataset_no_y

# calculate gradient descent 
def batch_gradient_descent(dataset,T,weight,r):

    # Initialize the norm weight difference |W_t - W_{t-1}| to some value greater than threshold
    norm_diff = 1.00

    # until total error is below threshold
    while norm_diff > T:
        pdf_jw = []
        for i in range(len(weight)):
            sum_pdfjw  = 0
            for row in dataset:
                sum_pdfjw -= grad_Jw(row[-1],weight, row[:7], row[i])
            pdf_jw.append(sum_pdfjw/len(dataset))

        # update weight
        updated_weight = [update_weight(weight[i],r,pdf_jw[i]) for i in range(len(weight))]
        
        # weight_difference
        w_diff = [updated_weight[t] - weight[t] for t in range(len(weight))] 
        norm_diff = np.sqrt(np.sum(np.square(w_diff)))

        # update the weight vector after all errors are calculated
        weight = updated_weight[:]

        # store new cost function
        costfunc_list.append(cost_func(weight,dataset))

        # update learning rate
        r /= 2

    return r, weight, costfunc_list

# calculate stochastic gradient descent 
def stochastic_gradient_descent(dataset,T,weight,r):

    # Initialize the norm weight difference |W_t - W_{t-1}| to some value greater than threshold
    norm_diff = 1.00

    # until total error is below threshold
    while norm_diff > T:
        pdf_jw = []
        for i in range(len(weight)):
            sum_pdfjw  = 0
            random_choice = random.randrange(0,len(data))
            sum_pdfjw -= grad_Jw(dataset[random_choice][-1],weight, dataset[random_choice], dataset[random_choice][i])
            pdf_jw.append(sum_pdfjw)

        # update weight
        updated_weight = [update_weight(weight[i],r,pdf_jw[i]) for i in range(len(weight))]
        
        # weight_difference
        w_diff = [updated_weight[t] - weight[t] for t in range(len(weight))] 
        norm_diff = np.sqrt(np.sum(np.square(w_diff)))

        # update the weight vector after all errors are calculated
        weight = updated_weight[:]

        # store new cost function
        costfunc_list.append(cost_func(weight,dataset))

        # update learning rate
        r /= 2

    return r, weight, costfunc_list

# batch gradient descent algorithm
def grad_descent(dataset,name):

    # initialising elements
    weight = np.zeros(7)     # initial weight, w_0
    threshold = 1/1000000       # tolerance level
    r = 1             # learning rate
    global costfunc_list 
    costfunc_list = []  # cost function values

    # Get initial cost_function value
    costfunc_list.append(cost_func(weight, dataset))

    # gradient descent to get next_weight, i.o.w w^(t+1)
    if name.lower() == "batch":
        r, weight, costfunc_list = batch_gradient_descent(dataset,threshold,weight,r)

    if name.lower() == "stochastic":
        r, weight, costfunc_list = stochastic_gradient_descent(dataset,threshold,weight,r)
    
    return r, weight, costfunc_list

# run
def main():
    ## BATCH GRADIENT DESCENT ##
    # empty arrays for train and test datasets
    dataset_train_raw = []
    dataset_test_raw = []

    # import train dataset
    with open('drive/MyDrive/concrete/train.csv', 'r') as train_file:
        for line in train_file:
            terms = line.strip().split(',')
            dataset_train_raw.append(terms)
    
    # import test dataset
    with open('drive/MyDrive/concrete/test.csv', 'r') as test_file:
        for line in test_file:
            terms = line.strip().split(',')
            dataset_test_raw.append(terms)
    
    dataset_train = [] 
    dataset_test = [] 
    # converting the datatype of each entry into float      
    for row in dataset_train_raw:
        row = np.asarray(row,dtype=float)
        dataset_train.append(row)
    for row in dataset_test_raw:
        row = np.asarray(row,dtype=float)
        dataset_test.append(row)

    # get actual label values
    # labels_val = [row[-1] for row in dataset_test]

    # get dataset without y value
    # dataset_train_no_y = dataset_no_y(dataset_train)
    # dataset_test_no_y = dataset_no_y(dataset_test)

    # batch gradient descent
    r, weight, costfunc_list = grad_descent(dataset_train, "batch")
    print("for batch gradient descent, r = ",r)
    print("for batch gradient descent, weight = ", weight)
    print("for batch gradient descent, cost function = ",costfunc_list)

    # plotting cost func at each steps
    plt.plot(costfunc_list)
    plt.xlabel('steps')
    plt.ylabel('Cost Function')
    plt.suptitle('Cost Functions at each steps for Batch Gradient Descent')
    plt.show()

    # report
    final_costfunc = cost_func(weight, dataset_test)
    final_r = r
    print("final cost function value for batch gradient descent = ", final_costfunc)
    print("learning rate for batch gradient descent = ", final_r)

    ## STOCHASTIC GRADIENT DESCENT ##
    # empty arrays for train and test datasets
    dataset_train_raw = []
    dataset_test_raw = []

    # import train dataset
    with open('drive/MyDrive/concrete/train.csv', 'r') as train_file:
        for line in train_file:
            terms = line.strip().split(',')
            dataset_train_raw.append(terms)
    
    # import test dataset
    with open('drive/MyDrive/concrete/test.csv', 'r') as test_file:
        for line in test_file:
            terms = line.strip().split(',')
            dataset_test_raw.append(terms)
    
    dataset_train = [] 
    dataset_test = [] 
    # converting the datatype of each entry into float      
    for row in dataset_train_raw:
        row = np.asarray(row,dtype=float)
        dataset_train.append(row)
    for row in dataset_test_raw:
        row = np.asarray(row,dtype=float)
        dataset_test.append(row)

    # get actual label values
    # labels_val = [row[-1] for row in dataset_test]

    # get dataset without y value
    # dataset_train_no_y = dataset_no_y(dataset_train)
    # dataset_test_no_y = dataset_no_y(dataset_test)

    # batch gradient descent
    r, weight, costfunc_list = grad_descent(dataset_train, "stochastic")
    print("for stochastic gradient descent, r = ",r)
    print("for stochastic gradient descent, weight = ", weight)
    print("for stochastic gradient descent, cost function = ",costfunc_list)

    # plotting cost func at each steps
    plt.plot(costfunc_list)
    plt.xlabel('steps')
    plt.ylabel('Cost Function')
    plt.suptitle('Cost Functions at each steps for Stochastic Gradient Descent')
    plt.show()

    # report
    final_costfunc = cost_func(weight, dataset_test)
    final_r = r
    print("final cost function value for stochastic gradient descent = ", final_costfunc)
    print("learning rate for stochastic gradient descent = ", final_r)

    # empty arrays for train and test datasets
    dataset_train_raw = []

    # import train dataset
    with open('drive/MyDrive/concrete/train.csv', 'r') as train_file:
        for line in train_file:
            terms = line.strip().split(',')
            dataset_train_raw.append(terms)

    dataset_train = [] 
    # converting the datatype of each entry into float      
    for row in dataset_train_raw:
        row = np.asarray(row,dtype=float)
        dataset_train.append(row)

    X = np.transpose(np.array([row[:-1] for row in dataset_train]))
    Y = np.array([row[-1] for row in dataset_train])

    weight_analysis = np.dot(np.linalg.inv(np.dot(X, np.transpose(X))), np.dot(X, Y))

    print("The analytical solution to the optimal weight vector is:" , weight_analysis)

if __name__ == "__main__":
    main()
