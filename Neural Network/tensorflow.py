import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import normalize
import numpy as np
import copy

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

    best_model = [0., 0., 0.]
    
    # relu activation
    for depth in [3, 5, 9]:
        for width in [5, 10, 25, 50, 100]:
            model = tf.keras.Sequential()
            
            # Add layers
            for _ in range(depth-1):
                model.add(tf.keras.layers.Dense(width, kernel_initializer=keras.initializers.he_normal(), activation='relu'))
        
            model.fit(dataset_train, y_label, epochs=10, verbose=0)
            
            evaluation = model.evaluate(dataset_train, y_label, verbose=0)
            accuracy = evaluation[1]
            error = 1 - accuracy
            print("Depth:", depth, "Width:", width, "Error:", error)
            
    # tanh activation
    for depth in [3, 5, 9]:
        for width in [5, 10, 25, 50, 100]:
            model = tf.keras.Sequential()

            # Add layers
            for _ in range(depth-1):
                model.add(Dense(1, activation=tf.keras.activations.tanh(x), kernel_initializer=keras.initializers.glorot_normal()))

            model.fit(dataset_train, y_label_test, epochs=10, verbose=0)

            evaluation = model.evaluate(dataset_train, y_label, verbose=0)
            accuracy = evaluation[1]
            error = 1 - accuracy
            print("Depth:", depth, "Width:", width, "Error:", error)

if __name__ == "__main__":
    main()
