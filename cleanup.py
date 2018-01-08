#
# andre@corp.insite.com.br
# 2017-10-15
#
# The goal here is to convert ratings.csv and movies.csv so that they remove unused films
# it takes the ratings.csv file and the movies.csv file on the current directory
# it saves the following files:
#   - converted-ratings-train.csv
#   - converted-ratings-test.csv 
#   - converted-movies.csv,  where movieids are cleaned up
#   - movies.tsv, with film data that can be used by tensorboard
#

import pandas as pd
import numpy as np
import collections
import re

# Criate dummies for each genre
create_genre_dummies = False

# expunge all films with less than this number of ratings:
min_ratings = 5

print("loading movies.csv...")
movies = pd.read_csv("movies.csv", sep=",")
# movieId,title,genres

print("loading ratings.csv...")
ratings = pd.read_csv("ratings.csv", sep=",")
# userId,movieId,rating,timestamp

print("Getting aggregate data")
print("Counting ratings per film")
num_ratings = ratings.groupby(['movieId'])['rating'].size().reset_index(name='num_ratings')
movies = pd.merge(movies, num_ratings, on='movieId').fillna(0)
ratings = pd.merge(ratings, num_ratings, on='movieId').fillna(0)

print("Calculating mean rating per film")
mean = ratings.groupby(['movieId'])['rating'].mean().reset_index(name='mean_ratings')
movies = pd.merge(movies, mean, on='movieId')
movies.columns = ( 'old_movieId', 'title', 'genres', 'num_ratings', 'mean_ratings')

print("renaming column movieId to old_movieId")
ratings.columns = ('userId', 'old_movieId', 'rating', 'timestamp' , 'num_ratings')

orig_rating_count = ratings.shape[0]
print("removing films with less than {} ratings".format(min_ratings))

ratings = ratings.loc[ratings['num_ratings'] >= min_ratings]
print("removing films with less than 5 ratings: {} left out of {}".format(ratings.shape[0], orig_rating_count))

movieids = pd.unique(ratings['old_movieId'])
print("Finding unique movieIds in ratings.csv: {}".format(movieids.size))

print("Creating new moviemap")
moviemap = collections.OrderedDict(zip(np.sort(movieids), range(1, movieids.size + 1)))
print("  Moviemap created: has size {}".format(len(moviemap.keys())))

print("Adding column movieId to ratings object with the renumbered films")
ratings = ratings.assign(movieId = ratings['old_movieId'].map(moviemap))

num_train_movies = movieids.size * 0.9
num_train_ratings = ratings.shape[0] * 0.9
print("start to separate training and test sets. Test set: around {} itens {} movies".format(num_train_ratings, num_train_movies))
all_train_ratings = ratings.loc[ratings['userId'] % 10 > 0 ].sample(frac=1)
print("  Para treinamento e testes: 90%% dos userIds: %d" % (all_train_ratings.shape[0]))

spot = int(all_train_ratings.shape[0]*9/10)
train_ratings = all_train_ratings.iloc[0:spot]
print("  Para treinamento: %d" % train_ratings.shape[0])
validation_set_ratings = all_train_ratings.iloc[spot:]
print("  Para cross validation: %d" % (validation_set_ratings.shape[0]))
test_set_ratings = ratings.loc[ratings['userId'] % 10 == 0 ]
print("training ratings found: {} test ratings: {} validation ratings: {}".format(train_ratings.shape[0], validation_set_ratings.shape[0], test_set_ratings.shape[0]))

print("ratings is done")

print("Some movies haven't been used by ratings.csv, so we have to prepare for that")
print("Translating movieIds to new numbers")
movies = movies.assign(movieId = movies['old_movieId'].map(moviemap))
year_r = re.compile(r'\((\d+)\)[ "]*$')

def extract_year(title):
   match = year_r.search(title)
   if match:
      return int(match.group(1))
   else:
      return 1865

titles = movies['title']
years = list(map(lambda title: extract_year(title), titles))
movies = movies.assign(year = years)

genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western']

def intGenre(genres, genre):
  return list(map(lambda g: int(g.find(genre) >= 0), genres))


print("Writing clean-movies.tsv (has to have only active movies because tensorboard requires it)")
movies_to_write = movies.dropna(axis=0, how='any')
# ['old_movieId', 'title', 'genres', 'num_ratings', 'mean_ratings', 'movieId', 'year'],
movies_to_write.loc[-1] = [0, 'PLACEHOLDER', '', 0, 0.0, 0, 0]
movies_to_write.sort_values('movieId', inplace=True)

if (create_genre_dummies):
  genres = movies_to_write['genres']
  movies_to_write = movies_to_write.assign(Action = intGenre(genres, 'Action'))
  movies_to_write = movies_to_write.assign(Adventure = intGenre(genres, 'Adventure'))
  movies_to_write = movies_to_write.assign(Animation = intGenre(genres, 'Animation'))
  movies_to_write = movies_to_write.assign(Children = intGenre(genres, 'Children'))
  movies_to_write = movies_to_write.assign(Comedy = intGenre(genres, 'Comedy'))
  movies_to_write = movies_to_write.assign(Crime = intGenre(genres, 'Crime'))
  movies_to_write = movies_to_write.assign(Documentary = intGenre(genres, 'Documentary'))
  movies_to_write = movies_to_write.assign(Drama = intGenre(genres, 'Drama'))
  movies_to_write = movies_to_write.assign(Fantasy = intGenre(genres, 'Fantasy'))
  movies_to_write = movies_to_write.assign(FilmNoir = intGenre(genres, 'Film-Noir'))
  movies_to_write = movies_to_write.assign(Horror = intGenre(genres, 'Horror'))
  movies_to_write = movies_to_write.assign(IMAX = intGenre(genres, 'IMAX'))
  movies_to_write = movies_to_write.assign(Musical = intGenre(genres, 'Musical'))
  movies_to_write = movies_to_write.assign(Mystery = intGenre(genres, 'Mystery'))
  movies_to_write = movies_to_write.assign(Romance = intGenre(genres, 'Romance'))
  movies_to_write = movies_to_write.assign(SciFi = intGenre(genres, 'Sci-Fi'))
  movies_to_write = movies_to_write.assign(Thriller = intGenre(genres, 'Thriller'))
  movies_to_write = movies_to_write.assign(War = intGenre(genres, 'War'))
  movies_to_write = movies_to_write.assign(Western = intGenre(genres, 'Western'))

#movies.assign('movieId', movies.astype(int))
movies_to_write.to_csv('clean-movies-' + str(min_ratings) + '.tsv', sep="\t", columns = ['movieId', 'title', 'genres', 'year', 'mean_ratings', 'num_ratings'], header = True, index = False)

print("Finding movies in movie.csv that haven't an assignment yet")
missing_movies = movies.loc[movies['movieId'].isnull(), 'old_movieId']
print("  missing_movies: {}".format(missing_movies.shape[0]))
n = max(moviemap.values()) + 1
print("  base for new numbers will be {}".format(n))

d = dict(zip(missing_movies, range(n, n + missing_movies.size + 1)))

moviemap.update(d)

movies = movies.assign(movieId = movies['old_movieId'].map(moviemap))
print("  Moviemap recreated: has size {}".format(len(moviemap.keys())))

print("Writing clean-movies-" + str(min_ratings) + ".csv")
movies.to_csv("clean-movies-" + str(min_ratings) + ".csv", sep=',', columns = ['movieId', 'old_movieId', 'title', 'genres', 'year', 'mean_ratings', 'min_ratings'], header = True, index = False)

print("Writing clean-ratings-" + str(min_ratings) + ".csv")
ratings.to_csv("clean-ratings-" + str(min_ratings) + ".csv", sep=',', columns = ['userId', 'movieId', 'rating', 'timestamp'], header = True, index = False)
print("  Writing train-ratings-" + str(min_ratings) + ".csv")
train_ratings.to_csv('train-ratings-' + str(min_ratings) + '.csv', sep=',', columns = ['userId', 'movieId', 'rating', 'timestamp'], header = True, index = False)
print("  Writing validation-ratings-" + str(min_ratings) + ".csv")
validation_set_ratings.to_csv('validation-ratings-' + str(min_ratings) + '.csv', sep=',', columns = ['userId', 'movieId', 'rating', 'timestamp'], header = True, index = False)
print("  Writing test-ratings-" + str(min_ratings) + ".csv")
test_set_ratings.to_csv('test-ratings-' + str(min_ratings) + '.csv', sep=',', columns = ['userId', 'movieId', 'rating', 'timestamp'], header = True, index = False)

