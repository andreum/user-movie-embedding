#!python

import tensorflow as tf
import pandas as pd
import numpy as np

print("Loading...")
print("Tensorflow: version %s" %(tf.__version__))
print("Pandas: version %s" %(pd.__version__))
print("Numpy: version %s" %(np.__version__))

DIMENSIONS = 1
USERS = 247753

print("1. Criando user_prefs")
user_prefs = tf.get_variable("user_prefs", [USERS, DIMENSIONS])
factor = tf.get_variable("factor", [DIMENSIONS])

print("2. Loading ratings") 

ratings = pd.read_csv("ratings.csv")

