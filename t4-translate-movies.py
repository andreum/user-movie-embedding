#

# andre@corp.insite.com.br
# 2017-10-08
#

#
# renormaliza os ids de filmes
# 
# usage: <csv file> 
#

import sys

fname = sys.argv[1]

f = open(fname, "r")
outf = open(fname + "-translated", "w")
mapf = open(fname + ".map", "w")

head = f.readline()
print("header: %s" % (head), end='')
print("from,to", file=mapf)
print("userId,movieId,rating", file=outf)

numitems = 0
map=dict()

for line in f:
  userid, movieid, rating, timestamp = line.split(',')
  if (movieid in map):
    #print("movieid {0} found in map. now it is {1}".format(movieid, new_movie_id))
    new_movie_id = map[movieid]
  else:
    new_movie_id = numitems
    map[movieid] = new_movie_id
    print(",".join(str(x) for x in [movieid, new_movie_id]), file=mapf)
    print('movieid {0} not found in map. now it\'s {1}'.format(movieid, new_movie_id))
    numitems = numitems+1
  print(','.join(str(x) for x in [int(userid) - 1, new_movie_id, rating]), file=outf)

