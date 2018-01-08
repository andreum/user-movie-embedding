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
from keras.model import Sequential
from keras.layers import Dense

from tensorflow.python import debug as tf_debug

NUM_USERS = 247754
NUM_MOVIES = 151712
batch_size = 99999
num_steps = 2000001
base_alpha = 0.1
count =1
# Regularization
lbda = 0.002
decay = 0.9999
num_ratings = 0
TRAIN_INPUT_FILE="train-ratings-3.csv"
TEST_INPUT_FILE="validation-ratings-3.csv"
use_bias = True
use_square = True
normalize = False
tf.set_random_seed(1)
round_ranking = 0
clip_output = True

t0 = time.perf_counter()

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

if len(sys.argv) < 4:
   use_activation = 'linear'
#   raise Exception('parameter activation is required. It can be "linear", "sigmoid" or "relu"')
else:
   if sys.argv[3] in ("sigmoid" , "linear", "lrelu", "relu"):
      use_activation = sys.argv[3]

if use_activation == 'sigmoid':
  scale = 6.0
else:
  scale = 1.0

def loga(msg):
  now = time.perf_counter()
  print("%6.2f: %s" % (now - t0, msg))

def load_data(train_fname, test_fname):
  global NUM_USERS
  global NUM_MOVIES
  global round_ranking
  global num_ratings
  print("Loading data from {} and {}".format(train_fname, test_fname))
  full_train_data = pd.read_csv(train_fname, sep=",").sample(frac=1)
  full_test_data = pd.read_csv(test_fname, sep=",").sample(frac=1)
  train_data = np.array(full_train_data[["userId", "movieId"]])
  train_labels = np.array(full_train_data[["rating"]])
  test_data = np.array(full_test_data[["userId", "movieId"]])
  test_labels = np.array(full_test_data[["rating"]])
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
  return train_data, train_labels, test_data, test_labels

def apply_activation(x):
  if (use_activation) == 'sigmoid':
    return tf.sigmoid(x) * scale
  elif use_activation == 'relu':
    return tf.nn.relu(x)
  elif use_activation == 'lrelu':
    return tf.maximum(x, 0.2 * x)
  else:
    return x

def variable_summaries(var, varname):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(varname):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    #with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


loga("feature: using {0} activation with scale {1}".format(use_activation, scale))
if use_activation == 'sigmoid':
  activation_str = "sigmoid_{}".format(scale)
else:
  activation_str = use_activation
  base_alpha = base_alpha / 10.0

i = 1
while (os.path.isdir("t9a-r{0:d}-bias{1:d}-L2{2:d}-f{3}-a{4}-round{5}-clip{6}-normalize{7}-{8}".format(NUM_FEATURES, int(use_bias), int(use_square), TRAIN_INPUT_FILE, activation_str, round_ranking, int(clip_output), int(normalize), i))):
   i = i + 1
dirname = "t9a-r{0:d}-bias{1:d}-L2{2:d}-f{3}-a{4}-round{5}-clip{6}-normalize{7}-{8}".format(NUM_FEATURES, int(use_bias), int(use_square), TRAIN_INPUT_FILE, activation_str, round_ranking, int(clip_output), int(normalize), i)
os.mkdir(dirname)
prefix = dirname + "/"
sys.stdout = open(prefix + "out", "w", 1)
graph = tf.Graph()

loga("feature: using {} activation".format(activation_str))

train_data, train_labels, test_data, test_labels = load_data(TRAIN_INPUT_FILE, TEST_INPUT_FILE)

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
      with tf.name_scope('embeddings'):
         user_embeddings = tf.get_variable("user_embeddings", [NUM_USERS, NUM_FEATURES], initializer=tf.random_normal_initializer(0,1*math.sqrt(1/NUM_FEATURES)))
         movie_embeddings = tf.get_variable("movie_embeddings", [NUM_MOVIES, NUM_FEATURES], initializer=tf.random_normal_initializer(0,1*math.sqrt(1/NUM_FEATURES)))
         variable_summaries(user_embeddings, 'user_embeddings')
         variable_summaries(movie_embeddings, 'movie_embeddings')
         
      with tf.name_scope('local_embeddings'):
         tf_user_embeddings = tf.gather(user_embeddings, tf_train_data[:,0])
         tf_movie_embeddings = tf.gather(movie_embeddings, tf_train_data[:,1])
         variable_summaries(tf_user_embeddings, 'tf_user_embeddings')
         variable_summaries(tf_movie_embeddings, 'tf_movie_embeddings')
   else:
      user_embeddings = tf.get_variable("user_embeddings", initializer = tf.constant(0.0))
      movie_embeddings = tf.get_variable("movie_embeddings", initializer = tf.constant(0.0))
   if normalize:
     unorm = tf.sqrt(tf.reduce_sum(tf.square(tf_user_embeddings), 1, keep_dims=True))
     tf_user_embeddings = tf_user_embeddings / unorm
	 # There's o need to regularize the biases
   movie_embeddings = tf.abs(movie_embeddings)
   #bias = tf.get_variable("bias", dtype=tf.float32, initializer=tf.constant(3.5))
   user_bias = tf.get_variable("user_bias", [NUM_USERS, 1], initializer=tf.random_normal_initializer(0.0))
   movie_bias = tf.get_variable("movie_bias", [NUM_MOVIES, 1], initializer=tf.random_normal_initializer(3.1))
   tf_user_bias = tf.gather(user_bias, tf_train_data[:,0])
   tf_movie_bias = tf.gather(movie_bias, tf_train_data[:,1])
   with tf.name_scope("biases"):
      variable_summaries(user_bias, 'user_bias')
      variable_summaries(movie_bias, 'movie_bias')
      variable_summaries(tf_user_bias, 'tf_user_bias')
      variable_summaries(tf_movie_bias, 'tf_movie_bias')

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
      tf_movie_bias

   if (clip_output):
      train_prediction = tf.minimum(5.0, tf.maximum(0.5, apply_activation(train_prediction)))
      loga("feature: clipping output")
   if  use_bias:
	   loga("feature: using biases")
   else:
	   loga("feature: NOT using biases")
   with tf.name_scope('results'):
      error = tf.subtract(train_prediction, tf_train_labels)
      sse = tf.reduce_sum(tf.square(error))
      tf.summary.scalar("sse", sse)
      if (NUM_FEATURES > 0):
         if (use_square):
            loga("feature: using L2 on movie embedding regularization")
            regularization = tf.reduce_sum(tf.square(tf_user_embeddings))/NUM_FEATURES/tf_batch_size + tf.reduce_sum(tf.square(tf_movie_embeddings))/NUM_FEATURES/tf_batch_size
         else:
            loga("feature: using L1 on movie embedding regularization")
            regularization = tf.reduce_sum(tf.square(tf_user_embeddings))/NUM_FEATURES/tf_batch_size + tf.reduce_sum(tf.abs(tf_movie_embeddings))/NUM_FEATURES/tf_batch_size
      else:
         regularization = tf.reduce_sum(tf.square(tf_movie_bias)) + tf.reduce_sum(tf.square(tf_user_bias))
         # + tf.reduce_sum(tf.square(tf_movie_bias))*batch_size/NUM_MOVIES + tf.reduce_sum(tf.square(tf_user_bias)) * batch_size / NUM_USERS
      loss = sse + lbda * regularization
      tf.summary.scalar("loss", loss)
      mse = sse / tf_batch_size
      tf.summary.scalar("batch_size", tf_batch_size)
      tf.summary.scalar("mse", mse)
   optimizer = tf.train.GradientDescentOptimizer(tf_lr).minimize(loss)
   #optimizer = tf.train.AdamOptimizer(tf_lr, epsilon=0.1).minimize(loss)
   histogram = tf.histogram_fixed_width(error, [-4.5, 4.5], nbins=10)
   tf.summary.histogram('error', error)
   merged = tf.summary.merge_all()
   saver = tf.train.Saver()

train_writer = tf.summary.FileWriter(dirname + "/train", graph)
validation_writer = tf.summary.FileWriter(dirname + "/validation", graph)


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
   pace = 1.01
   for step in range(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_data[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels, tf_lr: lr}
      if (step % display_interval == 0):
         _, l, predictions, uemb, memb, _mse, hist, ubias, mbias, summary = session.run(
            [optimizer, loss, train_prediction, user_embeddings, movie_embeddings, mse, histogram, user_bias, movie_bias, merged], feed_dict=feed_dict)
      else:
         _, l, predictions, uemb, memb, _mse, hist, ubias, mbias = session.run(
            [optimizer, loss, train_prediction, user_embeddings, movie_embeddings, mse, histogram, user_bias, movie_bias], feed_dict=feed_dict)

      acccount = acccount * decay + 1
      acctot = acctot * decay + l / batch_size
      accmse = accmse * decay + _mse
      exploss = acctot/acccount
      expmse = accmse/acccount

      if (step % display_interval  == 0):
         # TODO: make a loop to test all the itens and get the loss
         epoch = epoch + 1
         train_writer.add_summary(summary, epoch)
         feed_dict = {tf_train_data : test_data, tf_train_labels : test_labels, tf_lr: lr}
         test_l, test_predictions, test_mse, test_hist, val_summary = session.run(
            [loss, train_prediction, mse, histogram, merged], feed_dict=feed_dict)
         validation_writer.add_summary(val_summary, epoch)

         if (test_l / old_loss -1 >= -1e-6):
            lr = lr * 0.2
            batch_size = int(batch_size  * 0.7) + 300
            print("    batch_size now is {}".format(batch_size))
            decay = 1.0 - (batch_size/num_ratings)
            display_interval = int(num_ratings / batch_size)
            #pace = 1.0
         else:
            lr = lr * pace
         old_loss = test_l
	 #
         loga("Minibatch loss at epoch %d step %d: %f (%f): (%f) on test set: (%f)" % (epoch, step, l, exploss, l/batch_size, test_l))
         print("  Mean Square Error: %.10f - exp=%.9f - test mse=%.9f" % (_mse, expmse, test_mse))
         print("  Learning Rate: %.10f batch_size=%d" % (lr, batch_size))
         if (NUM_FEATURES > 0):
            print("  user embeddings: %f: %s" % (np.linalg.norm(uemb)/math.sqrt(uemb.size), np.mean(uemb, 0)))
            print("  movie embeddings: %f: %s" % (np.linalg.norm(memb)/math.sqrt(memb.size), np.mean(memb, 0)))
         print("  user bias: %f: %f" % (np.linalg.norm(ubias)/math.sqrt(ubias.size), np.mean(ubias, 0)))
         print("  movie bias: %f: %f" % (np.linalg.norm(mbias)/math.sqrt(mbias.size), np.mean(mbias, 0)))
         #print("bias: %f" % (_bias))
         print("  error: %s" % (hist))
         print("  test:  %s" % (test_hist))
         #print("user embeddings: %f" % (user_embeddings))
         #print("embeddings: {}".format(emb))
         #valid_prediction.eval(), valid_labels))
         #print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
         if (epoch % 20 == 0):
            print("saving model to {}model.ckpt".format(prefix))
            saver.save(session, prefix + "model.ckpt")
            print("saving... done")

         if lr < 1e-10:
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

