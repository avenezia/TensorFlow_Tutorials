import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
# train is an Operation that updates the variables in var_list argument of minimize;
# if var_list is None it defaults to the list of variables collected in the graph.
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # initialize values to have a big initial loss.
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})
  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
  print("Iteration %s: W: %s b: %s loss: %s"%(i, curr_W, curr_b, curr_loss))

# evaluate training accuracy
# Run method returns the first argument (fetches) with the udated values.
final_W, final_b, final_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("Final model: W: %s b: %s loss: %s"%(final_W, final_b, final_loss))