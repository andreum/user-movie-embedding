#
# andre@corp.insite.com.br
# 2017-10-10
# Codigo que faz regressao simples e encontra embeddings
#
# a ideia aqui e a seguinte:
#  - carregar dados do movielens
#  - inicializar o embedding de forma aleatoria
#  -  encontrar os embeddings de filmes e de usuarios que gerem o menor erro possivel
# t8: retira os bias de filmes e usuarios e substitui por um unico bias global

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
batch_size = 9999
num_steps = 2000001
base_alpha = 0.001
count =1
# Regularization
lbda = 0.00001
decay = 0.9999
INPUT_FILE="clean-ratings.csv"
use_bias = False
use_square = True
use_activation = 'sigmoid'
if use_activation == 'sigmoid':
  scale = 6.0
else:
  scale = 1.0
tf.set_random_seed(1)
round_ranking = 0
use_intermediate_activation = 1
use_second_layer = 1

if use_bias and use_second_layer:
  print("second layer nao é compativel com biases no momento, setando para use_bias para 0")
  use_bias = 0

t0 = time.perf_counter()

def loga(msg):
  now = time.perf_counter()
  print("%6.2f: %s" % (now - t0, msg))

def load_data(fname):
  global NUM_USERS
  global NUM_MOVIES
  global round_ranking
  print("Loading data from {}".format(fname))
  full_train_data = pd.read_csv(INPUT_FILE, sep=",").sample(frac=1)
  train_data = np.array(full_train_data[["userId", "movieId"]])
  train_labels = np.array(full_train_data[["rating"]])
  if (round_ranking):
    train_labels = np.floor(train_labels)
  NUM_USERS = np.amax(train_data[:,0]) + 1
  NUM_MOVIES = np.amax(train_data[:,1]) + 1
  num_ratings = train_data.shape[0]
  loga("NUM_USERS = {}".format(NUM_USERS))
  loga("NUM_MOVIES = {}".format(NUM_MOVIES))
  loga("num ratings = {}".format(num_ratings))
  loga("batch_size = {}".format(batch_size))
  loga("num_steps = {}".format(num_steps))
  loga("uses second layer: {}".format(bool(use_second_layer)))
  loga("uses intermediate activation: {}".format(bool(use_intermediate_activation)))
  return train_data, train_labels

def apply_activation(x):
  if (use_activation) == 'sigmoid':
    return tf.sigmoid(x)
  elif use_activation == 'relu':
    return tf.nn.relu(x)
  elif use_activation == 'lrelu':
    return tf.maximum(x, 0.2 * x)
  else:
    return x

if sys.argv[1].isdigit():
   NUM_FEATURES = int(sys.argv[1])
else:
   raise Exception("parameter NUM_FEATURES is required")

if len(sys.argv) < 3:
   raise Exception("parameter round_ranking is required (y, Y, s, S, 1, T, t means should round down. Anything else means it shouldn't")

if sys.argv[2] in ("y", "Y", "s", "S", "1", "T", "t"):
   round_ranking = 1
else:
   round_raking = 0

loga("feature: using {0} activation with scale {1}".format(use_activation, scale))
if use_activation == 'sigmoid':
  activation_str = "sigmoid_{}".format(scale)
  if use_second_layer:
    base_alpha = base_alpha / NUM_FEATURES
else:
  activation_str = use_activation
  base_alpha = base_alpha / 10.0

i = 1
while (os.path.isdir("t10-r{0:d}-bias{1:d}-L2{2:d}-f{3}-a{4}-round{5}-layer2_{6}-{7}".format(NUM_FEATURES, int(use_bias), int(use_square), INPUT_FILE, activation_str, round_ranking, use_second_layer, i))):
   i = i + 1
dirname = "t10-r{0:d}-bias{1:d}-L2{2:d}-f{3}-a{4}-round{5}-layer2_{6}-{7}".format(NUM_FEATURES, int(use_bias), int(use_square), INPUT_FILE, activation_str, round_ranking, use_second_layer, i)
os.mkdir(dirname)
prefix = dirname + "/"
sys.stdout = open(prefix + "out", "w", 1)

loga("feature: using {} activation".format(activation_str))

train_data, train_labels = load_data(INPUT_FILE)

graph = tf.Graph()
with graph.as_default():
   tf_train_data = tf.placeholder(tf.int32, shape=(batch_size, 2))
   tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 1))
   tf_lr = tf.placeholder(tf.float32)
   print("when setting graph: NUM_USERS: {}".format(NUM_USERS))
   print("when setting graph: NUM_MOVIES: {}".format(NUM_MOVIES))

   tf_count = tf.get_variable("count", dtype=tf.int32, initializer=tf.constant(count))
   if (NUM_FEATURES > 0):
      ones = tf.constant(1., shape=(NUM_FEATURES,1))
      user_embeddings = tf.get_variable("user_embeddings", [NUM_USERS, NUM_FEATURES], initializer=tf.random_normal_initializer(0,1*math.sqrt(1/NUM_FEATURES)))
      movie_embeddings = tf.get_variable("movie_embeddings", [NUM_MOVIES, NUM_FEATURES], initializer=tf.random_normal_initializer(0,1*math.sqrt(1/NUM_FEATURES)))
      tf_user_embeddings = tf.gather(user_embeddings, tf_train_data[:,0])
      tf_movie_embeddings = tf.gather(movie_embeddings, tf_train_data[:,1])
   else:
      user_embeddings = tf.get_variable("user_embeddings", initializer = tf.constant(0.0))
      movie_embeddings = tf.get_variable("movie_embeddings", initializer = tf.constant(0.0))
   if use_second_layer:
      weights = tf.get_variable("weights", [NUM_FEATURES, 1], initializer=tf.random_normal_initializer(0, 1.0 * math.sqrt(1/NUM_FEATURES)))
   #movie_embeddings = tf.abs(movie_embeddings)
   #bias = tf.get_variable("bias", dtype=tf.float32, initializer=tf.constant(3.5))
   user_bias = tf.get_variable("user_bias", [NUM_USERS, 1], initializer=tf.random_normal_initializer(0.0))
   movie_bias = tf.get_variable("movie_bias", [NUM_MOVIES, 1], initializer=tf.random_normal_initializer(3.5))
   tf_user_bias = tf.gather(user_bias, tf_train_data[:,0])
   tf_movie_bias = tf.gather(movie_bias, tf_train_data[:,1])

   #train_prediction = tf.tensordot(tf_user_embeddings, tf_movie_embeddings, axes=1)
   if (NUM_FEATURES > 0):
      if (use_bias):
         train_prediction = apply_activation(tf.matmul(tf.multiply(tf_user_embeddings, tf_movie_embeddings), ones) + tf_user_bias + tf_movie_bias)
      else:
         if use_second_layer:
            train_prediction = apply_activation(tf.matmul(tf.nn.sigmoid(tf.multiply(tf_user_embeddings, tf_movie_embeddings)), weights)) * scale
         elif use_intermediate_activation:
            train_prediction = apply_activation(tf.matmul(apply_activation(tf.multiply(tf_user_embeddings, tf_movie_embeddings)), ones))
         else:
            train_prediction = apply_activation(tf.matmul(tf.multiply(tf_user_embeddings, tf_movie_embeddings), ones))
         
      #train_prediction = tf.matmul(tf.multiply(tf_user_embeddings, tf_movie_embeddings), ones) + tf_movie_bias + bias
   else:
      #train_prediction = tf_user_bias + tf_movie_bias + bias 
      #train_prediction = 5.0 * tf.sigmoid(tf_user_bias + tf_movie_bias)
      train_prediction = apply_activation(tf_movie_bias)
   if  use_bias:
	   loga("feature: using biases")
   else:
	   loga("feature: NOT using biases")
   error = tf.subtract(train_prediction, tf_train_labels)
   sse = tf.reduce_sum(tf.square(error))
   if (NUM_FEATURES > 0):
      if (use_square):
         loga("feature: using L2 on movie embedding regularization")
         regularization = tf.reduce_sum(tf.square(tf_user_embeddings))/NUM_FEATURES/batch_size + tf.reduce_sum(tf.square(tf_movie_embeddings))/NUM_FEATURES/batch_size
      else:
         loga("feature: using L1 on movie embedding regularization")
         regularization = tf.reduce_sum(tf.square(tf_user_embeddings))/NUM_FEATURES + tf.reduce_sum(tf.abs(tf_movie_embeddings))/NUM_FEATURES
   else:
      regularization = tf.reduce_sum(tf.square(tf_movie_bias)) + tf.reduce_sum(tf.square(tf_user_bias))
	# There's o need to regularize the biases
	# + tf.reduce_sum(tf.square(tf_movie_bias))*batch_size/NUM_MOVIES + tf.reduce_sum(tf.square(tf_user_bias)) * batch_size / NUM_USERS
   if use_second_layer:
      if use_square:
         regularization = regularization + tf.reduce_sum(tf.square(weights))/NUM_FEATURES
      else:
         regularization = regularization + tf.reduce_sum(tf.abs(weights))/NUM_FEATURES
   loss = sse + lbda * regularization
   mse = sse / batch_size
   optimizer = tf.train.GradientDescentOptimizer(tf_lr).minimize(loss)
   histogram = tf.histogram_fixed_width(error, [-4.5, 4.5], nbins=10)
   saver = tf.train.Saver()


with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print("Initialized")
   uemb, memb = session.run([user_embeddings, movie_embeddings])
   print("user embeddings: {}\n",uemb)
   print("movie embeddings: {}\n",memb)
   acccount = acctot = 0.0
   old_loss = 1e20
   lr = base_alpha
   for step in range(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_data[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels, tf_lr: lr}
      if (use_second_layer):
         _, l, predictions, uemb, memb, _mse, hist, ubias, mbias, w = session.run(
            [optimizer, loss, train_prediction, user_embeddings, movie_embeddings, mse, histogram, user_bias, movie_bias, weights], feed_dict=feed_dict)
      else:
         _, l, predictions, uemb, memb, _mse, hist, ubias, mbias = session.run(
            [optimizer, loss, train_prediction, user_embeddings, movie_embeddings, mse, histogram, user_bias, movie_bias], feed_dict=feed_dict)
      acccount = acccount * 0.9999 + 1
      acctot = acctot * 0.9999 + _mse
      exploss = acctot/acccount
      if (step % 2000  == 0):
         if (exploss > old_loss):
            lr = lr * 0.3
         else:
            lr = lr * 1.05
         old_loss = exploss
	 #
         loga("Minibatch loss at step %d: %f (%f)" % (step, l, l/batch_size))
         print("  Mean Square Error: %f - exp=%10f" % (_mse, acctot/acccount))
         print("  Learning Rate: %f" % (lr))
         if (NUM_FEATURES > 0):
            print("  user embeddings: %f: %s" % (np.linalg.norm(uemb)/uemb.size, np.mean(uemb, 0)))
            print("  movie embeddings: %f: %s" % (np.linalg.norm(memb)/memb.size, np.mean(memb, 0)))
         print("  user bias: %f: %f" % (np.linalg.norm(ubias)/ubias.size, np.mean(ubias, 0)))
         print("  movie bias: %f: %f" % (np.linalg.norm(mbias)/mbias.size, np.mean(mbias, 0)))
         if use_second_layer:
            print("  weight: %f: %f: %s" % (np.linalg.norm(w)/w.size, np.mean(w, 0), np.transpose(w)))
         #print("bias: %f" % (_bias))
         print("  error: %s" % (hist))
         #print("user embeddings: %f" % (user_embeddings))
         #print("embeddings: {}".format(emb))
         #valid_prediction.eval(), valid_labels))
         #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
         if (step % 60000 == 0):
            print("saving model to {}model.ckpt".format(prefix))
            saver.save(session, prefix + "model.ckpt")
            print("saving... done")

         if lr < 1e-14:
            break
   print("ENDED. Steps done: {}".format(step))
   print("saving model to {}model.ckpt".format(prefix))
   saver.save(session, prefix + "model.ckpt", global_step=step)
   if (NUM_FEATURES > 0):
      print("user_embeddings:\n{}".format(np.around(uemb, 3)))
      print("movie_embeddings:\n{}".format(np.around(memb, 3)))
      np.savetxt(prefix + "user_embeddings.csv.gz", uemb, delimiter=',', fmt="%.7f")
      np.savetxt(prefix + "movie_embeddings.csv.gz", memb, delimiter=',', fmt="%.7f")
   else:
      print("NO EMBEDDINGS")
   np.savetxt(prefix + "user_bias.csv.gz", ubias, delimiter=',', fmt="%.7f")
   np.savetxt(prefix + "movie_bias.csv.gz", mbias, delimiter=',', fmt="%.7f")

