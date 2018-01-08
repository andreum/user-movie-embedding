#
# andre@corp.insite.com.br
# 2017-10-10
# Codigo que faz regressao simples e encontra embeddings
#
# a ideia aqui e a seguinte:
#  - carregar dados do movielens
#  - inicializar o embedding de forma aleatoria
#  -  encontrar os embeddings de filmes e de usuarios que gerem o menor erro possivel
#

from __future__ import division
from __future__ import print_function
from time import gmtime, strftime, localtime

import math
import time
import sys
import os
#from pylab import *
from scipy import sparse
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from tensorflow.python import debug as tf_debug

NUM_USERS = 247754
NUM_MOVIES = 151712
NUM_FEATURES = 1
batch_size = 99998
num_steps = 20001
lbda = 5.0 * NUM_FEATURES
alpha = 0.1
decay = 0.99999
INPUT_FILE="ratings-100000.csv"
prefix = "t6-r{0:d}-l{1}-a{2}-{3}-".format(NUM_FEATURES, lbda, alpha, INPUT_FILE)

tf.set_random_seed(1)

t0 = time.perf_counter()

def loga(msg):
  now = time.perf_counter()
  print("%6.2f: %s" % (now - t0, msg))

def gendata(num):
  for i in range(num):
    a = random.randrange(0, NUM_USERS)
    b = random.randrange(0, NUM_USERS)
    c = a + b
    yield a, b, c

def load_data(fname):
  print("Loading data from {}".format(fname))
  full_train_data = pd.read_csv(INPUT_FILE, sep=",").sample(frac=1)
  train_data = np.array(full_train_data[["userId", "movieId"]])
  train_labels = np.array(full_train_data[["rating"]])
  return train_data, train_labels

graph = tf.Graph()
with graph.as_default():
   ones = tf.constant(1., shape=(NUM_FEATURES,1))
   user_embeddings = tf.get_variable("user_embeddings", [NUM_USERS, NUM_FEATURES], initializer=tf.constant_initializer(2*math.sqrt(1/NUM_FEATURES)))
   movie_embeddings = tf.get_variable("movie_embeddings", [NUM_MOVIES, NUM_FEATURES], initializer=tf.constant_initializer(2*math.sqrt(1/NUM_FEATURES)))
   tf_train_data = tf.placeholder(tf.int32, shape=(batch_size, 2))
   tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 1))
   tf_user_embeddings = tf.gather(user_embeddings, tf_train_data[:,0])
   tf_movie_embeddings = tf.gather(movie_embeddings, tf_train_data[:,1])

   #train_prediction = tf.tensordot(tf_user_embeddings, tf_movie_embeddings, axes=1)
   train_prediction = tf.matmul(tf.multiply(tf_user_embeddings, tf_movie_embeddings), ones)
   error = tf.subtract(train_prediction, tf_train_labels)
   sse = tf.reduce_sum(tf.square(error))
   loss = sse + alpha * tf.reduce_sum(tf.square(tf_user_embeddings))/NUM_FEATURES + alpha * tf.reduce_sum(tf.square(tf_movie_embeddings))/NUM_FEATURES
   mse = sse / batch_size
   lbda = lbda * decay
   optimizer = tf.train.GradientDescentOptimizer(lbda/batch_size/NUM_FEATURES).minimize(loss)
   histogram = tf.histogram_fixed_width(error, [-4.5, 4.5], nbins=10)

train_data, train_labels = load_data("ratings-100000.csv")

with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print("Initialized")
   uemb, memb = session.run([user_embeddings, movie_embeddings])
   print("user embeddings: {}\n",uemb)
   print("movie embeddings: {}\n",memb)
   acccount = acctot = 0.0
   for step in range(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_data[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}
      _, l, predictions, uemb, memb, _mse, hist = session.run(
            [optimizer, loss, train_prediction, user_embeddings, movie_embeddings, mse, histogram], feed_dict=feed_dict)
      acctot = acctot * 0.99 + _mse
      acccount = acccount * 0.99 + 1.0
      if (step % 100  == 0):
         loga("Minibatch loss at step %d: %f (%f)" % (step, l, l/batch_size))
         print("  Mean Square Error: %f" % (_mse))
         print("  Accuracy: %f" % (acctot / acccount))
         print("user embeddings: %f" % (np.linalg.norm(uemb)))
         print("movie embeddings: %f" % (np.linalg.norm(memb)))
         print("error: %s" % (hist))
         #print("user embeddings: %f" % (user_embeddings))
         #print("embeddings: {}".format(emb))
         #print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
         #print("Validation accuracy: %.1f%%" % accuracy(
         #valid_prediction.eval(), valid_labels))
         #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
   print("user_embeddings:\n{}".format(np.around(uemb, 3)))
   print("movie_embeddings:\n{}".format(np.around(memb, 3)))
   np.savetxt(prefix + "user_embeddings.csv.gz", uemb, delimiter=',', fmt="%.5f")
   np.savetxt(prefix + "movie_embeddings.csv.gz", memb, delimiter=',', fmt="%.5f")
   
