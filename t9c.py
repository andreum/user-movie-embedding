#!/bin/env python3 -u

#
# Tests results of regression from t9a.py
# andre@corp.insite.com.br - 2017-11-06
#
# parameters: t9c.py <logdir>
#

import pandas as pd
import numpy as np
import argparse
import os
import time
import math
import sys

parser = argparse.ArgumentParser(description='Test results of movielens regression.')
parser.add_argument('logdir', nargs=1, help='directory with data from the tensorflow run')
parser.add_argument('--ratings', nargs=1, help='file containing ratings to be tested', required = True)
parser.add_argument('--movies', nargs=1, help='file containing movie info', required = True)
parser.add_argument('--csvout', nargs=1, help='output file with test results', required = True)


args = parser.parse_args()
logdir = args.logdir[0]
print("Directory: {}".format(logdir))

sys.stdout = open(logdir + "/validation.out", "w", 1)
sys.stderr = sys.stdout

ratfile = args.ratings[0]
print("Ratings File: {}".format(ratfile))
csvout = args.csvout[0]
print("Output CSV file: {}".format(csvout))
movies_file = args.movies[0]
print("Movies file: {}".format(movies_file))


if "linear" in logdir:
  print("Linear activation")
  t_regression = "linear"
elif "asigmoid" in logdir:
  print("sigmoid activation")
  t_regression = "sigmoid"
else:
  print("Hmmm. directory doesn't have a recognized type of regression (sigmoid or linear)")
  sys.exit(1)

if not os.path.exists(logdir):
  print("Diretory doesn't exist")
  sys.exit(2)

if not os.path.isdir(logdir):
  print("{} is not a directory".format(logdir))
  sys.exit(3)

if not os.path.exists(ratfile):
  print("{} does not exist")
  sys.exit(4)


t0 = time.perf_counter()
def loga(msg):
  now = time.perf_counter()
  print("%6.2f: %s" % (now - t0, msg))

memb = pd.read_csv(logdir + "/movie_embeddings.csv.gz", header = None)
loga("Movie embeddings: {}".format(memb.shape))
uemb = pd.read_csv(logdir + "/user_embeddings.csv.gz", header = None)
loga("User embeddings: {}".format(uemb.shape))
ubias = pd.read_csv(logdir + "/user_bias.csv.gz", header=None)
loga("User Bias: {}".format(ubias.shape))
mbias = pd.read_csv(logdir + "/movie_bias.csv.gz", header=None)
loga("Movie Bias: {}".format(mbias.shape))
movies = pd.read_csv(movies_file)
loga("Movies: {}".format(movies.shape))

loga("Loading ratings...")
ratings = pd.read_csv(ratfile)
loga("Ratings: {}".format(ratings.shape))
userIds = np.sort(ratings['userId'].unique())
loga("Unique users: {}".format(userIds.shape))
csvfname = logdir + "/" + csvout
print("opening csv output file: {}".format(csvfname))
outf=open(csvfname, "w", buffering = 1)

num_features = uemb.shape[1]
mean_ratings = movies['mean_ratings']

print('"context","num_movies","mean_error","mse"', file=outf)
#loga("{0},{1},{2:.3f},{3:.3f}".format(i, num_movies, 
#      (np.sum(validation_predicted_score) - np.sum(validation_actual_score))/num_movies, 
#       np.sum(np.square(validation_predicted_score - validation_actual_score))/num_movies), file=outf)
for userId in userIds:
  loga("==== userId: {}".format(userId))
  user_ratings = ratings.loc[ratings['userId'] == userId]
  user_movieIds = user_ratings['movieId'].values
  predicted_ratings = movies.loc[user_movieIds,]['mean_ratings'].values
  actual_ratings = user_ratings['rating'].values
  diff = actual_ratings - predicted_ratings
  print("diffs: {}".format(diff))

  old_user_vector = None
  #np.random.shuffle(user_ratings)

  validation_movieIds = user_ratings['movieId'].values
  num_movies = validation_movieIds.shape[0]
  print("{0},{1},{2:.3f},{3:.3f}".format(0, num_movies, 
       np.mean(diff),
       np.mean(np.square(diff))), file=outf)

  for i in range(1, user_ratings.shape[0]):
    seen_movieIds = user_ratings[0:i]['movieId'].values
    validation_movieIds = user_ratings[i:]['movieId'].values
    # NUM_FEATURES x n
    #seen_actual_score = user_ratings[0:i]['rating'].values
    seen_actual_score = np.matrix(user_ratings[0:i]['rating']).T
    # TODO: precisa testar isto...
    seen_memb = memb.loc[seen_movieIds,] # (n, NUM_FEATURES)
    # loga("seen_movie embeddings: {}".format(seen_memb))
    seen_movie_bias = mbias.loc[seen_movieIds].values
    #loga("DEBUG: seen_movie_bias: {} ({})".format(seen_movie_bias, seen_movie_bias.shape))
    inversora = np.linalg.pinv(seen_memb)
    # loga("DEBUG: inverter matrix: {}".format(inversora))
    score_offset = seen_actual_score - seen_movie_bias
    # loga("DEBUG: score offset: [{}] ({})".format(score_offset.T, score_offset.shape))

    user_vector = np.matmul(inversora, score_offset)
    seen_user_bias = (score_offset - np.matmul(seen_memb, user_vector)).mean()
    if  i == 1:
      rotation = 0
    else:
      loga("user_vector shapes: {} and {}".format(old_user_vector.shape, user_vector.shape))
      rotation = np.matmul(np.transpose(old_user_vector), user_vector)/np.linalg.norm(old_user_vector)/np.linalg.norm(user_vector)
      if num_features > 1:
         try:
           loga(" change in user vector: {}: {}: norm: {} to {}".format(rotation, math.acos(rotation)*180/math.pi, np.linalg.norm(old_user_vector), np.linalg.norm(user_vector)))
         except:
           loga("Unexpected error:", sys.exc_info()[0])
           loga("{0:f} {1} {2}".format(rotation, old_user_vector, user_vector))

    old_user_vector = user_vector
      
    loga("User vector: {} ({}) [{}]".format(user_vector.T, user_vector.shape, np.linalg.norm(user_vector)))
    #loga("DEBUG: shapes: {}, {}".format(np.matmul(seen_memb, user_vector).shape, seen_movie_bias.shape))
    #loga("DEBUG: {}, {}".format(np.matmul(seen_memb, user_vector), seen_movie_bias))
    seen_predicted_score = np.add(np.matmul(seen_memb, user_vector), seen_movie_bias)
    seen_predicted_score = np.minimum(np.maximum(0.5, seen_predicted_score + seen_user_bias), 5.0)
    loga("  user bias:   {}".format(seen_user_bias))
    #loga("  predicted score: {}".format(predicted_score))
    #loga("  actual scores:   {}".format(seen_actual_score))
    loga("  fixed: context: {0}  mse: {2:.3f}".format(i, (np.sum(seen_predicted_score) - np.sum(seen_actual_score))/i, np.sum(np.square(seen_predicted_score - seen_actual_score))/i))
 
    validation_memb = memb.loc[validation_movieIds,].values
    validation_movie_bias = mbias.loc[validation_movieIds].values
    validation_predicted_score = np.minimum(5.0,np.maximum(0.5,np.add(np.add(np.matmul(validation_memb, user_vector), validation_movie_bias), seen_user_bias)))
    validation_actual_score = np.matrix(user_ratings[i:]['rating']).T
    loga("  predicted: {} {}[t]".format(validation_predicted_score.shape, np.transpose(validation_predicted_score)))
    loga("  actual:    {} {}[t]".format(validation_actual_score.shape, validation_actual_score.T))
    validation_error = validation_actual_score - validation_predicted_score
    loga("  error:     {} {}".format(validation_error.shape, validation_error.T))
    num_movies = validation_movieIds.shape[0]
    loga("  context: {0}  num elements: {1}  avg error: {2:.3f}   mse: {3:.3f}".format(i, num_movies, 
      (np.sum(validation_predicted_score) - np.sum(validation_actual_score))/num_movies, 
       np.sum(np.square(validation_predicted_score - validation_actual_score))/num_movies))
    print("{0},{1},{2:.3f},{3:.3f}".format(i, num_movies, 
      (np.sum(validation_predicted_score) - np.sum(validation_actual_score))/num_movies, 
       np.sum(np.square(validation_predicted_score - validation_actual_score))/num_movies), file=outf)
    loga("---")
    

