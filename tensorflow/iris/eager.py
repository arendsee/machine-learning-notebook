# The following code is derived from the tutorial:
# https://www.tensorflow.org/get_started/eager
#
# The one change I've made is functionalizing everything

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from os.path import expanduser

def initialize():
  tf.enable_eager_execution()


def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)
  # First 4 fields are features, combine into single tensor
  features = tf.reshape(parsed_line[:-1], shape=(4,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label


def get_train_dataset():
  train_dataset_fp = expanduser("~/.keras/datasets/iris_training.csv")
  if(not os.path.exists(train_dataset_fp)):
    # Download Iris data set to cache directory ($HOME/.keras/datasets/iris_training.csv) 
    train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
    # train_dataset_fp is the filepath to the iris dataset
    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                               origin=train_dataset_url)
  # set parameters for the training dataset
  train_dataset = tf.data.TextLineDataset(train_dataset_fp)
  train_dataset = train_dataset.skip(1)          # skip the first header row
  train_dataset = train_dataset.map(parse_csv)   # parse each row
  train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
  train_dataset = train_dataset.batch(32)
  return train_dataset


def get_test_dataset():
  test_fp = expanduser("~/.keras/datasets/iris_test.csv")
  if(not os.path.exists(test_fp)):
    test_url = "http://download.tensorflow.org/data/iris_test.csv"
    test_fp = tf.keras.utils.get_file(
      fname=os.path.basename(test_url),
      origin=test_url
    )

  test_dataset = tf.data.TextLineDataset(test_fp)
  test_dataset = test_dataset.skip(1)             # skip header row
  test_dataset = test_dataset.map(parse_csv)      # parse each row with the funcition created earlier
  test_dataset = test_dataset.shuffle(1000)       # randomize
  test_dataset = test_dataset.batch(32)           # use the same batch size as the training set

  return test_dataset


def make_model(depth, bounds):
  layers = [
    tf.keras.layers.Dense(
      depth[0], # number of nodes in first layer
      activation="relu",
      input_shape=(bounds[0],)
    )
  ] + [
    tf.keras.layers.Dense(
       d, # number of nodes in ith layer
       activation="relu"
    ) for d in depth[1:-1]
  ] + [
    tf.keras.layers.Dense(bounds[1])
  ]
  return tf.keras.Sequential(layers)


def make_loss():
  def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
  return loss


def make_grad(loss):
  def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
      loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)
  return grad


def make_optimizer(learning_rate=0.01):
  # Choose and parameterize the optimizing algorithm
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  return optimizer


def train_model(train_dataset, model, grad, optimizer, num_epochs=201):
  # Train the model
  for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    # Training loop - using batches of 32
    for x, y in tfe.Iterator(train_dataset):
      # Optimize the model
      grads = grad(model, x, y)
      optimizer.apply_gradients(zip(grads, model.variables),
                                global_step=tf.train.get_or_create_global_step())
  return model 


def test_model(test_dataset, model):
  test_accuracy = tfe.metrics.Accuracy()
  for (x, y) in tfe.Iterator(test_dataset):
    prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)
  return test_accuracy


def print_prediction(predictions, class_ids):
  for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    name = class_ids[class_idx]
    print("{}\t{}".format(i, name))


if __name__ == '__main__':
  initialize()
  train_d = get_train_dataset()
  test_d = get_test_dataset()

  # These are the three main 
  depth = [10,10]
  num_epochs = 1000

  model = make_model(depth, bounds=(4,3))

  model = train_model(
    train_dataset = train_d,
    model         = model,
    grad          = make_grad(make_loss()),
    optimizer     = make_optimizer(),
    num_epochs    = num_epochs
  )
  print("Test set accuracy: {:.3%}".format(test_model(test_d, model).result()))


  example_d = tf.convert_to_tensor([
      [5.1, 3.3, 1.7, 0.5,],
      [5.9, 3.0, 4.2, 1.5,],
      [6.9, 3.1, 5.4, 2.1]
    ])
  class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]

  pred = model(example_d)
  print_prediction(pred, class_ids)
