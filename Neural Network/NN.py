import numpy as np
import math
import copy
import random

# Sigmoid activation function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
  return x * (1 - x)

# Back-propagation algorithm to compute the gradient with respect to all the edge weights
# given one training example
def ann_backpropagation(X, labels, weights, biases):
  # Forward pass
  # Activations of layer 0 (input layer)
  a0 = X 
  # z's to layer 1
  z1 = np.dot(a0, weights[0]) + biases[0] 
  z1_1 = z1[0][0]
  z1_2 = z1[0][1] 
  # print(weights[0])
  # Activations of layer 1 (hidden layer)
  a1 = sigmoid(z1)  
  # z's to layer 3
  z2 = np.dot(a1, weights[1]) + biases[1] 
  z2_1 = z2[0][0]
  z2_2 = z2[0][1]
  
  # Activations of layer 2 
  a2 = sigmoid(z2) 
  print(weights[2])

  # output layer = y
  y = np.dot(a2, weights[2]) + biases[2]
  # Backward pass
  # Compute the error at the output layer
  dL_dy = y - labels
  dL_dw3_b = np.dot(dL_dy, 1)
  dL_dw3 = np.dot(dL_dy, a2)

  # Compute the gradient for the weights and biases at the output layer
  dL_dw3_b = np.dot(dL_dy, 1)
  dL_dw3 = np.dot(dL_dy, a2)

  # Compute the gradient for the weights and biases at the hidden layer
  dL_dz2 = np.dot(dL_dy,weights[2].transpose())
  dL_dw2_b = dL_dz2 * sigmoid_derivative(a2)
  dL_dw2 = dL_dz2 * a1.transpose() * sigmoid_derivative(a2)

  # Compute the gradient for the weights and biases at the input layer
  dL_dz1 = np.dot(dL_dz2 * sigmoid_derivative(a2), weights[1].transpose())
  dL_dw1_b = dL_dz1 * sigmoid_derivative(a1)
  dL_dw1 = dL_dz1 * X.transpose() * sigmoid_derivative(a1)

  # Return the gradients for the weights and biases as a tuple
  print("dL_dw3", dL_dw3)
  print("dL_dw2", dL_dw2)
  print("dL_dw1", dL_dw1)
  print("dL_dw3_b", dL_dw3_b) 
  print("dL_dw2_b", dL_dw2_b)
  print("dL_dw1_b", dL_dw1_b)

def gaussian_distr_prior(w_i,v):
  return 1/np.sqrt(2*np.pi*v[i]) * np.exp(-np.dot(w_i, w_i.T)/2*v)

def stochastic_logistic_regression(dataset_train_with_bias, y, lr, alpha, v):
  # initialise w
  w = np.zeros(4)
  epoch = np.arange(1,100, step = 1, dtype = int)
  counter = 0

  for t in epoch:

    # random shuffle the examples

    for i, x in enumerate(dataset_train_with_bias):
      g_j = ( (math.exp(3 * y[i] * np.dot(w.transpose(),x)) * (-y[i] * x)) / (1 + math.exp(-y[i] * np.dot(w.transpose(),x))) ) + w/v
      gj_lst.append(g_j)

      w = w - lr * g_j

      lr = update_lr(lr, alpha, counter)
      counter += 1

  return w

def update_lr(int_lr, alpha,t):
  lr = int_lr/(1+(int_lr/alpha)*t)
  return lr

def gaussian_distr(w_i):
  return 1/np.sqrt(2*np.pi) * np.exp(-np.dot(w_i, w_i.T)/2)

def weights_init(width):

  layers=[5, width, width, 1]

  w = {}
  b = {}

  for i in range(len(layers)-1):

      layer_num = i + 1
      input_size = layers[i]
      output_size = layers[i + 1]

      w[layer_num] = gaussian_distr(np.random.randn(output_size, input_size))
      b[layer_num] = gaussian_distr(np.random.randn(output_size, 1))

  return list(w.values()),list(b.values())

def stochastic_gd_ANN(dataset_train_with_bias, y_label, T, lr, alpha, width):
  # length of the dataset
  N = float(len(dataset_train_with_bias))

  # randomise the initial weight based on gaussian distribution
  w, b = weights_init(width)

  curr_lr = lr
  counter = 1
  w[-1] = 1

  for i in range(T):

      # shuffle the data
      # np.random.shuffle(dataset_train_with_bias)

      s_idx = np.random.permutation(dataset_train_with_bias)
    
      for i,x_i in enumerate(dataset_train_with_bias) :
          # compute derivative of the loss function w.r.t weight for each example
          dL_dw_i = ann_backpropagation(x_i, y_label[i], w, b)

          # update the weight
          w = w - curr_lr * dL_dw_i
          
          # update the learning rate
          curr_lr = update_lr(lr, alpha, counter)
          counter += 1
  return w

def main():

  # load csv dataset into numpy array
  dataset_train = np.loadtxt('drive/MyDrive/ML_data/bank-note-2/train.csv', delimiter=',')
  dataset_test = np.loadtxt('drive/MyDrive/ML_data/bank-note-2/test.csv', delimiter=',')

  # making dataset with bias
  dataset_train_with_bias = []
  dataset_train_cp = copy.deepcopy(dataset_train)
  for x in dataset_train_cp:
    x[-1] = 1
    dataset_train_with_bias.append(x)

  dataset_train_with_bias_tmp = copy.deepcopy(dataset_train_with_bias)

  # seperating the label from training dataset
  y_label = []
  dataset_train_cp2 = copy.deepcopy(dataset_train)
  for x in dataset_train_cp2:
    y_label.append(x[-1])
  for i,y in enumerate(y_label):
    if y == 0:
      y_label[i] = -1

# seperating the label from test dataset
  y_label_test = []
  dataset_test_cp2 = copy.deepcopy(dataset_test)
  for x in dataset_test_cp2:
    y_label_test.append(x[-1])
  for i,y in enumerate(y_label_test):
    if y == 0:
      y_label_test[i] = -1
  
  # inputs for backpropagation question 2A
  weights_q2a = np.array([[-2,2],[-3,3]]), np.array([[-2,2],[-3,3]]), np.array([[2],[-1.5]]) 
  biases = np.array([-1, 1]), np.array([-1, 1]), np.array([-1])
  X = np.array([[1,1]])

  # learning rate
  lr =  0.01
  alpha = 0.01

  # backpropagation application based on q3 on written part
  ann_backpropagation(X, labels, weights_q2a, biases)

  # stochastic gradient descent on neural network with weights randomised on gaussian distribution
  weight_width = {}
  width = [5,10,25,50,100]
  for w in width:
    weight_width[w] = stochastic_gd_ANN(dataset_train_with_bias, y_label, 100, lr , alpha, w)
  
  # stochastic gradient descent on logistic regression
  weight_variances = {}
  variances = [0.01,0.1,0.5,1,3,5,10,100]
  for v in variances:
    weight_variances[v] = stochastic_logistic_regression(dataset_train_with_bias, y, lr, alpha, v)

# RESULTS:
# dL_dw3 [[-0.061967   -3.37492823]]
# dL_dw2 [[-0.00030092  0.00022569]
#  [-0.12139856  0.09104892]]
# dL_dw1 [[0.00105061 0.00157591]
#  [0.00105061 0.00157591]]
# dL_dw3_b [[-3.43689523]]
# dL_dw2_b [[-0.12169947  0.09127461]]
# dL_dw1_b [[0.00105061 0.00157591]]

# weights initialistion for stochastic nn
# weights
# ([array([[3.08453942e-02, 7.12371729e-02, 3.87305906e-01, 2.24933548e-02,
#           9.29761093e-03],
#          [7.12371729e-02, 4.13197582e-02, 6.55237671e-01, 1.73509389e-01,
#           3.40864739e-02],
#          [3.87305906e-01, 6.55237671e-01, 6.87184629e-02, 3.82244677e-02,
#           7.78002317e-01],
#          [2.24933548e-02, 1.73509389e-01, 3.82244677e-02, 4.38565668e-04,
#           1.81266682e-02],
#          [9.29761093e-03, 3.40864739e-02, 7.78002317e-01, 1.81266682e-02,
#           4.78372181e-04]]),
#   array([[2.14305547e-01, 4.39132717e-01, 6.06696683e-01, 6.40639032e-01,
#           2.45564095e-01],
#          [4.39132717e-01, 9.73758471e-03, 1.34771980e+00, 1.16819561e-02,
#           7.05935532e-01],
#          [6.06696683e-01, 1.34771980e+00, 3.63789285e-02, 1.63853652e-01,
#           6.67080675e-02],
#          [6.40639032e-01, 1.16819561e-02, 1.63853652e-01, 2.95440242e-04,
#           4.63158256e-02],
#          [2.45564095e-01, 7.05935532e-01, 6.67080675e-02, 4.63158256e-02,
#           1.61314942e-02]]),
#   array([[0.02522274]])],
# bias
#  [array([[0.39579021, 0.44122512, 0.40945964, 0.41800226, 0.43188104],
#          [0.44122512, 0.11099479, 0.28667641, 0.22055068, 0.14566618],
#          [0.40945964, 0.28667641, 0.36630102, 0.34231114, 0.30752955],
#          [0.41800226, 0.22055068, 0.34231114, 0.30315246, 0.25015052],
#          [0.43188104, 0.14566618, 0.30752955, 0.25015052, 0.18043889]]),
#   array([[0.39627653, 0.43506978, 0.41108955, 0.41297722, 0.37546725],
#          [0.43506978, 0.13004875, 0.27069337, 0.2551239 , 0.87390346],
#          [0.41108955, 0.27069337, 0.34884542, 0.34176828, 0.52328748],
#          [0.41297722, 0.2551239 , 0.34176828, 0.33378814, 0.54542849],
#          [0.37546725, 0.87390346, 0.52328748, 0.54542849, 0.23049905]]),
#   array([[0.03906051]])])



# gj_lst = []
# w_lst = []
# w = np.zeros(5)
# # w = np.array([ 0.0025, -0.005 ,  0.0015,  0.005 ])
# w_lst.append(w)

# x_1 = np.array([0.5, -1, 0.3, 1])
# x_2 = np.array([-1, -2, -2, -1])
# x_3 = np.array([1.5, 0.2, -2.5, 1])
# ex = [x_1, x_2, x_3]

# y_1 = 1
# y_2 = -1
# y_3 = 1
# y= dataset_train[:,4]

# lr_1 = 0.01
# lr_2 = 0.005
# lr_3 = 0.0025
# lr = [lr_1, lr_2, lr_3]

# epoch = np.arange(10)

# N = len(dataset_train[0])

# for t in epoch:
#   for i, x in enumerate(dataset_train_with_bias):
#     g_j = ( (math.exp(N * y[i] * np.dot(w.transpose(),x)) * (-y[i] * x)) / (1 + math.exp(-y[i] * np.dot(w.transpose(),x))) ) + w
#     gj_lst.append(g_j)

#     w = w - update_lr(lr, alpha, counter) * g_j
#     w_lst.append(w)

# for i,w in enumerate(w_lst) :
#   print("w", i, w)

# for i, j, in enumerate(gj_lst):
#   print("gj", i, j)
