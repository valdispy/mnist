# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:55:09 2019

@author: valdis
"""
import numpy as np
import tensorflow as tf
import mnist_reader

def most_common_value(item): 
    return np.argmax(np.bincount(item))

def prediction_labels(X_train, Y_train, X_test):
    tf_Xtrain = tf.compat.v1.placeholder(shape = [train_count, train_len], dtype = tf.int32) 
    tf_Ytrain = tf.compat.v1.placeholder(shape = [train_count], dtype = tf.int32)
    tf_Xtest = tf.compat.v1.placeholder(shape = [num_per_chunk, train_len], dtype = tf.int32)
        
    distance_values = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(tf_Xtrain, tf.expand_dims(tf_Xtest, 1))), axis=2)
    _, first_k_indices = tf.nn.top_k(tf.negative(distance_values), k = k_value)
    first_k_label = tf.gather(tf_Ytrain, first_k_indices)
         
    sess = tf.compat.v1.Session()
    first_k_label = sess.run(first_k_label, feed_dict = {tf_Xtrain : X_train, tf_Ytrain : Y_train, tf_Xtest : X_test})
    return first_k_label

if __name__ == "__main__":
    
    k_value = 11; num_per_chunk = 10; chunk_index = 0
    X_train, Y_train = mnist_reader.load_mnist('data', kind='train')
    X_test, Y_test = mnist_reader.load_mnist('data', kind='t10k')
        
    test_count, _ = X_test.shape; train_count, train_len = X_train.shape
    num_chunks = int(test_count / num_per_chunk)
    X_test_chunks = np.array_split(X_test, num_chunks)
    
    predicted_labels = []
    for test_chunk in X_test_chunks:
        
        print(chunk_index + 1, '/', num_chunks)
        first_k_label = prediction_labels(X_train, Y_train, test_chunk)
        Y_test_predict = np.apply_along_axis(most_common_value, 1, first_k_label)
        predicted_labels.extend(Y_test_predict.tolist())
        chunk_index += 1
    
    num_of_equal = np.count_nonzero(Y_test == np.array(predicted_labels))
    accuracy_value = num_of_equal / test_count
    print('k_value =', k_value, '; accuracy_value =', accuracy_value)
    
