#
# andre@corp.insite.com.br
# 2017-10-10
# Codigo que faz regressao simples e encontra embeddings
#
# a ideia aqui é a seguinte:
#  - gerar 100 objetos cada um dos quais tem um embedding unidimensional
#  - inicializar o embedding de forma aleatoria
#  - encontrar dois numeros aleatorios de 0 a 99: a e b
#  - aplicar a seguinte forma: embedding[a] + embedding[b] = a + b
#  - o objetivo é encontrar os embeddings tais que embedding[i] = i
#

from __future__ import division
from __future__ import print_function
from time import gmtime, strftime, localtime

import time
import sys
import os
from pylab import *
from scipy import sparse
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from tensorflow.python import debug as tf_debug

NUM_ITEMS = 100
NUM_FEATURES = 1
batch_size = 100
NUM_SAMPLES = 10000

tf.set_random_seed(1)

t0 = time.perf_counter()

def loga(msg):
  now = time.perf_counter()
  print("%6.2f: %s" % (now - t0, msg))

def gendata(num):
  for i in range(num):
    a = random.randrange(0, NUM_ITEMS)
    b = random.randrange(0, NUM_ITEMS)
    c = a + b
    yield a, b, c

full_train_data = np.array([list(x) for x in gendata(NUM_SAMPLES)])
train_data = full_train_data[:,[0,1]]
train_labels = full_train_data[:,[2]]


graph = tf.Graph()
with graph.as_default():
   embeddings = tf.get_variable("embeddings", [NUM_ITEMS, NUM_FEATURES])
   tf_train_data = tf.placeholder(tf.int32, shape=(batch_size, 2))
   tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 1))
   train_prediction = tf.add(tf.gather(embeddings, tf_train_data[:,0]), tf.gather(embeddings, tf_train_data[:,1]))
   error = tf.subtract(train_prediction, tf_train_labels)
   loss = tf.reduce_sum(tf.matmul(error, error, transpose_a = True)) + .001 * tf.reduce_sum(tf.matmul(embeddings, embeddings, transpose_a = True))/batch_size
   optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)


num_steps = 3001
with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print("Initialized")
   for step in range(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_data[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}
      _, l, predictions, emb = session.run(
            [optimizer, loss, train_prediction, embeddings], feed_dict=feed_dict)
      if (step % 1000  == 0):
         print("Minibatch loss at step %d: %f" % (step, l))
         #print("embeddings: {}".format(emb))
         #print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
         #print("Validation accuracy: %.1f%%" % accuracy(
         #valid_prediction.eval(), valid_labels))
         #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
   print("embeddings: {}".format(np.around(emb, 2)))

