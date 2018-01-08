import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
result = sess.run(hello)

print(result)

