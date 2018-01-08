#
#
# andre@corp.insite.com.br
# 2017-10-07
#

#
# Convert csv file to feather format
# usage: <prefixo-arquivo>

import feather
import pandas as pd
import sys

prefix = sys.argv[1]

print("loading csv")
# userId,movieId,rating,timestamp
df = pd.read_csv(prefix + ".csv", usecols=['userId', 'movieId', 'rating'])
print("Columns: " + df.columns)
df.columns = ['user_id', 'item_id', 'rating']
#print("Columns to: " + df.columns)

print("writing feather")
feather.write_dataframe(df, prefix + '.feather')

print("Done")

