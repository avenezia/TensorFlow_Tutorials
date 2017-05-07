# coding= utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("MNIST has {0} training, {1} cross validation and {2} test samples".format(mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples))
print("MNIST training data shape " + str(mnist.train.images.shape))

# Being a placeholder means that it will be fed before asking TensorFlow to run a computation.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# The model.
y = tf.matmul(x, W) + b

# Placeholder for the correct labels.
y_ = tf.placeholder(tf.float32, [None, 10])

# The original formulation of cross-entropy:
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# with y = tf.nn.softmax(tf.matmul(x, W) + b)
# tf.reduce_sum adds the elements in the second dimension of y due to the reduction_indices=[1].
# tf.reduce_mean computes the mean over all the examples in the batch.
# This formulation can be numerically instable, so it is better to use the following
# tf.nn.softmax_cross_entropy_with_logits on the raw outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# The parameter of GradientDescentOptimizer is the learning rate α.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    # Using only a small random subset of the input data at each iteration represents the 
    # stochastic gradient descent.
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # Here we fill the placeholders.
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# tf.argmax gives the index of the highest entry in a tensor along some axis. 
# For example, tf.argmax(y,1) is the label our model thinks is most likely for each input,
# while tf.argmax(y_,1) is the correct label. We can use tf.equal to check if our prediction matches the truth.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# This gives a list of booleans. To determine what fraction are correct, we cast to floating point numbers and
# then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Accuracy on test data.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
