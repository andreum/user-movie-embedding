
# andre@corp.insite.com.br
# 2017-09-25
#
# testes com keras

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, Activation
from keras import optimizers
from keras.utils import plot_model

model = Sequential()
model.add(Embedding(1000,1, embeddings_initializer='uniform'))

# 
sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer = sgd, loss='mse')
plot_model(model, to_file='model.png', show_shapes=True)

