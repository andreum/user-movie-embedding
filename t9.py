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
batch_size = 99999
num_steps = 2000001
base_alpha = 0.003
count =1
# Regularization
lbda = 0.0001
decay = 0.9999
num_ratings = 0
INPUT_FILE="clean-ratings.csv"
use_bias = False
use_square = True
use_activation = 'linear'
if use_activation == 'sigmoid':
  scale = 6.0
else:
  scale = 1.0
tf.set_random_seed(1)
round_ranking = 0

t0 = time.perf_counter()

def loga(msg):
  now = time.perf_counter()
  print("%6.2f: %s" % (now - t0, msg))

def load_data(fname):
  global NUM_USERS
  global NUM_MOVIES
  global round_ranking
  global num_ratings
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
  return train_data, train_labels

def apply_activation(x):
  if (use_activation) == 'sigmoid':
    return tf.sigmoid(x) / scale
  elif use_activation == 'relu':
    return tf.nn.relu(x)
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
else:
  activation_str = use_activation
  base_alpha = base_alpha / 10.0

i = 1
while (os.path.isdir("t9-r{0:d}-bias{1:d}-L2{2:d}-f{3}-a{4}-round{5}-{6}".format(NUM_FEATURES, int(use_bias), int(use_square), INPUT_FILE, activation_str, round_ranking, i))):
   i = i + 1
dirname = "t9-r{0:d}-bias{1:d}-L2{2:d}-f{3}-a{4}-round{5}-{6}".format(NUM_FEATURES, int(use_bias), int(use_square), INPUT_FILE, activation_str, round_ranking, i)
os.mkdir(dirname)
prefix = dirname + "/"
sys.stdout = open(prefix + "out", "w", 1)

loga("feature: using {} activation".format(activation_str))

train_data, train_labels = load_data(INPUT_FILE)

graph = tf.Graph()
with graph.as_default():
   tf_train_data = tf.placeholder(tf.int32, shape=(None, 2))
   tf_train_labels = tf.placeholder(tf.float32, shape=(None, 1))
   tf_lr = tf.placeholder(tf.float32)
   tf_batch_size = tf.cast(tf.shape(tf_train_data)[0], tf.float32)
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
   movie_embeddings = tf.abs(movie_embeddings)
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
         regularization = tf.reduce_sum(tf.square(tf_user_embeddings))/NUM_FEATURES/tf_batch_size + tf.reduce_sum(tf.square(tf_movie_embeddings))/NUM_FEATURES/tf_batch_size
      else:
         loga("feature: using L1 on movie embedding regularization")
         regularization = tf.reduce_sum(tf.square(tf_user_embeddings))/NUM_FEATURES/tf_batch_size + tf.reduce_sum(tf.abs(tf_movie_embeddings))/NUM_FEATURES/tf_batch_size
   else:
      regularization = tf.reduce_sum(tf.square(tf_movie_bias)) + tf.reduce_sum(tf.square(tf_user_bias))
	# There's o need to regularize the biases
	# + tf.reduce_sum(tf.square(tf_movie_bias))*batch_size/NUM_MOVIES + tf.reduce_sum(tf.square(tf_user_bias)) * batch_size / NUM_USERS
   loss = sse + lbda * regularization
   mse = sse / tf_batch_size
   optimizer = tf.train.GradientDescentOptimizer(tf_lr).minimize(loss)
   histogram = tf.histogram_fixed_width(error, [-4.5, 4.5], nbins=10)
   saver = tf.train.Saver()


with tf.Session(graph=graph) as session:
   tf.global_variables_initializer().run()
   print("Initialized")
   uemb, memb = session.run([user_embeddings, movie_embeddings])
   print("user embeddings: {}\n",uemb)
   print("movie embeddings: {}\n",memb)
   acccount = acctot = accmse = 0.0
   old_loss = 1e20
   lr = base_alpha
   decay = 1.0 - (batch_size/num_ratings)
   display_interval = int(num_ratings / batch_size)
   epoch = 0
   for step in range(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_data[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels, tf_lr: lr}
      _, l, predictions, uemb, memb, _mse, hist, ubias, mbias = session.run(
            [optimizer, loss, train_prediction, user_embeddings, movie_embeddings, mse, histogram, user_bias, movie_bias], feed_dict=feed_dict)
      acccount = acccount * decay + 1
      acctot = acctot * decay + l / batch_size
      accmse = accmse * decay + _mse
      exploss = acctot/acccount
      expmse = accmse/acccount
      if (step % display_interval  == 0):
         epoch = epoch + 1
         if (exploss > old_loss):
            lr = lr * 0.2
            batch_size = int(batch_size  * 0.7) + 300
            decay = 1.0 - (batch_size/num_ratings)
            display_interval = int(num_ratings / batch_size)
         else:
            lr = lr * 1.02
         old_loss = exploss
	 #
         loga("Minibatch loss at epoch %d step %d: %f (%f): (%f)" % (epoch, step, l, exploss, l/batch_size))
         print("  Mean Square Error: %.10f - exp=%f" % (_mse, expmse))
         print("  Learning Rate: %f" % (lr))
         if (NUM_FEATURES > 0):
            print("  user embeddings: %f: %s" % (np.linalg.norm(uemb)/uemb.size, np.mean(uemb, 0)))
            print("  movie embeddings: %f: %s" % (np.linalg.norm(memb)/memb.size, np.mean(memb, 0)))
         print("  user bias: %f: %f" % (np.linalg.norm(ubias)/ubias.size, np.mean(ubias, 0)))
         print("  movie bias: %f: %f" % (np.linalg.norm(mbias)/mbias.size, np.mean(mbias, 0)))
         #print("bias: %f" % (_bias))
         print("  error: %s" % (hist))
         #print("user embeddings: %f" % (user_embeddings))
         #print("embeddings: {}".format(emb))
         #valid_prediction.eval(), valid_labels))
         #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
         if (epoch % 20 == 0):
            print("saving model to {}model.ckpt".format(prefix))
            saver.save(session, prefix + "model.ckpt")
            print("saving... done")

         if lr < 1e-11:
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

