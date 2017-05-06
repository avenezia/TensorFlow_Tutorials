from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("MNIST has {0} training, {1} cross validation and {2} test samples".format(mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples))
print("MNIST training data shape " + str(mnist.train.images.shape))