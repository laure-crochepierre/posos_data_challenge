import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df_input = pd.read_csv('DATA/input_train.csv', sep=";", index_col=0)
df_output = pd.read_csv('DATA/output_train.csv', sep=";", index_col=0)

features = df_input.columns
y_data = [x[0] for x in df_output.values]
X_data = [x[0] for x in df_input.values]


tfid_vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word',
                                  ngram_range =(1,3), stop_words={'french'})

X_preprocessed = tfid_vectorizer.fit_transform(X_data)

X_to_split, X_validation, y_to_split, y_validation = train_test_split(X_preprocessed, y_data, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_to_split, y_to_split, test_size=0.2, random_state=42)

n_hidden_1 = 10
n_hidden_2 = 5
n_input = total_words
classes = [x for x in y_data if x not in classes]
n_classes = len(classes)

def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1_activation = tf.nn.relu(layer_1_addition)

    #Hidden layer with relu activation
    layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2_activation = tf.nn.relu(layer_2_addition)

    #Output layer with linear activation
    out_layer_multiplication = tf.matmul(layer_2_activation, weights['ou'])
    out_layer_addition = out_layer_multiplication + biases['ou']
    return out_layer_addition 

init = tf.global_variables_initializer()

with tf.session() as sess:
    init.run()
    learning_rate = 0.001
    X = tf.placeholder(tf.float32, shape=[batch_size])
    y = tf.placeholder(tf.float32, shape=[batch_size,1])

    prediction = multilayer_perceptron(input_tensor, weights, biases)

    entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output)
    loss = tf.reduce_mean(entropy_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)<
 
