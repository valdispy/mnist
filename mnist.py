# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:55:09 2019

@author: valdis
"""
import mnist_reader
import tensorflow as tf
import numpy as np

def most_common_value(item): 
    return np.argmax(np.bincount(item))

if __name__ == "__main__":
        
    num_k = 3
    train_size = 3000; test_size = 300
    
    X_train, Y_train = mnist_reader.load_mnist('data', kind='train')
    X_test, Y_test = mnist_reader.load_mnist('data', kind='t10k')
    all_train_count, all_train_len = X_train.shape; all_test_count, _ = X_test.shape

    idx_train = np.random.randint(all_train_count, size=train_size)
    idx_test = np.random.randint(all_test_count, size=test_size)
           
    part_X_train = X_train[idx_train]; part_Y_train = Y_train[idx_train]   
    part_X_test = X_test[idx_test]; part_Y_test = Y_test[idx_test]  
    train_count, train_len = part_X_train.shape; test_count, _ = part_X_test.shape

    tf_Xtrain = tf.compat.v1.placeholder(shape = [train_count, train_len], dtype = tf.int32) 
    tf_Ytrain = tf.compat.v1.placeholder(shape = [train_count], dtype = tf.int32)    
    tf_Xtest = tf.compat.v1.placeholder(shape = [test_count, train_len], dtype = tf.int32)
     
    distance_values = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(tf_Xtrain, tf.expand_dims(tf_Xtest, 1))), axis=2)
    _, first_k_indices = tf.nn.top_k(tf.negative(distance_values), k = num_k)
    first_k_label = tf.gather(tf_Ytrain, first_k_indices)
     
    sess = tf.compat.v1.Session()
    first_k_label = sess.run(first_k_label, 
                             feed_dict = {tf_Xtrain : part_X_train, tf_Ytrain : part_Y_train, tf_Xtest: part_X_test})

    Y_test_predict = np.apply_along_axis(most_common_value, 1, first_k_label)    
    num_of_equal = np.count_nonzero(Y_test_predict == part_Y_test)
    
    accuracy_value = num_of_equal / len(part_Y_test)
    print('Accuracy :', accuracy_value)