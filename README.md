# user-movie-embedding
This project implements user and movie embeddings on top of the MovieLens database using Tensorflow/Keras

It was the result of a graduation project. More information can be obtained on the paper, but the gist is that we try to predict user preferences using matrix factorization as a way to generate user and movie embeddings. User embeddings assign a vector to each user and, likewise, Movie Embeddings assign a vector to each movie such that the movie vector will identify attributes while the user vector will give us information about user preferences for each of those attributes. In the end what we are doing is matrix factorization (minus some bias parameters), or factor analysis. Both are well known techniques, but this is still interesting, because here we can use movie and user vectors to make incremental estimates of user preferences based on a few ratings given by the user.. 

To use it, you first need to download the MovieLens dataset from https://grouplens.org/datasets/movielens/

Then you should use the cleanup.py file to preprocess the dataset. We use only the ratings.csv file then use the movies.csv for external validation.

t9.py is the code you should use for matrix factorization, and k?.py are the files that you can use if you prefer to use dense neural networks.


Andre Uratsuka Manoel

