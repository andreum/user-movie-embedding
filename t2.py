#!python

# andre@corp.insite.com.br
# 2017-10-07
#

#
# Objetivo é processar movieLens com tensorflow
#

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.DEBUG)

tf.logging.info("Loading...")
print("Tensorflow: version %s" %(tf.__version__))
print("Numpy: version %s" %(np.__version__))

# Constantes para a dimensao do problema
# numero de usuarios
USERS = 247753
# numero de filmes
MOVIES = 151711
# numero de dimensoes em que vamos mapear as preferencias
DIMENSIONS = 1

# user_prefs é a variável que da as preferencias dos usuarios, para cada usuario, quao forte é a preferencia dele para cada 
#  dimensao
print("1. Criando user_prefs") 
user_prefs = tf.get_variable('user_prefs', [USERS, DIMENSIONS])

# para cada filme, quanto que o filme traz de força em cada dimensão
# factor = tf.get_variable("factor", [DIMENSIONS])
movie_factors = tf.get_variable("movie_factors", [DIMENSIONS, MOVIES])
b = tf.Variable(tf.zeros([10]))

print("2. Loading ratings to memory") 

# userId,movieId,rating,timestamp
# 1,169,2.5,1204927694

filename_queue = tf.train.string_input_producer(["ratings-headerless.csv"])

reader = tf.TextLineReader()

key, value = reader.read(filename_queue)

record_defaults = [[], [], [], []]
userId, movieId, rating, timestamp = tf.decode_csv(value, record_defaults = record_defaults)

# Dimensions: userId, MovieId, rating, timestamp
#entrydata = tf.placeholder(tf.float32, [None, 4])

def model_calc(userId, movieId, rating):
  iuserId = tf.to_int32(userId)
  imovieId = tf.to_int32(movieId)
  # rating: 0 to 10
  rating = tf.to_int32(rating * 2)
  
  # This is a vector of user preferences
  prefs = tf.gather(user_prefs, iuserId)

  # This is a vector of user strengths
  strengths = tf.gather(movie_factors, movieId)
  return tf.nn.softmax(tf.matmul(prefs, strengths) + b)

pr = tf.Print(rating, [key, value, userId, movieId, rating], message="line entry: [key, value, userId, movieId, rating] = ", first_n = 1000, name="printer")

features = tf.stack([userId, movieId])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    # example, label = sess.run([features, rating])
    example, label = sess.run([features, pr])

  coord.request_stop()
  coord.join(threads)

