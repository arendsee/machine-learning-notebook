# The following code is adapted from the tutorial:
# https://www.tensorflow.org/get_started/eager
#
# I've adapted it to my orphan dataset

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from os.path import expanduser

def parse_orphan_dataset(line):
  record_defaults = [
    [1 ], # ps - phylostatum level 
    [1.], # length
    [1.], # exon.count
    [1.], # pI 
    [1.], # masked
    [1.], # GC
    [1.], # transmembrane.domains
    [1.], # ncomp
    [1.], # coils 
    [1.], # rem465
    [1.]  # hotloops
  ]
  n_features = len(record_defaults) - 1
  parsed_line = tf.decode_csv(
    line,
    record_defaults = record_defaults
  )
  # Combine features into single tensor
  features = tf.reshape(parsed_line[1:], shape=(n_features,))
  # Get labels 
  label = tf.reshape(parsed_line[0] - 1, shape=())
  return features, label


def get_train_dataset(training_fp):
  # set parameters for the training dataset
  train_dataset = tf.data.TextLineDataset(training_fp)
  train_dataset = train_dataset.skip(1)          # skip the first header row
  train_dataset = train_dataset.map(parse_orphan_dataset)   # parse each row
  train_dataset = train_dataset.shuffle(buffer_size=99999)  # randomize
  train_dataset = train_dataset.batch(32)
  return train_dataset


def get_test_dataset(testing_fp):
  test_dataset = tf.data.TextLineDataset(testing_fp)
  test_dataset = test_dataset.skip(1)          # skip the first header row
  test_dataset = test_dataset.map(parse_orphan_dataset)
  test_dataset = test_dataset.shuffle(99999)      # randomize
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


def train_model(train_dataset, model, grad, loss, optimizer, num_epochs=10):
  train_loss_results = []
  train_accuracy_results = []

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
      # Track progress
      epoch_loss_avg(loss(model, x, y))  # add current batch loss
      # compare predicted label to actual label
      epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 10 == 0:
      msg = "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}"
      print(msg.format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

  return (model, train_accuracy_results, train_loss_results)


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

  tf.enable_eager_execution()
  train_d = get_train_dataset("train.csv")
  test_d = get_test_dataset("test.csv")

  depth = [20,20,20]
  num_epochs = 1000

  model = make_model(depth, bounds=(10,19))

  loss = make_loss()

  model, train_accuracy, train_loss = train_model(
    train_dataset = train_d,
    model         = model,
    grad          = make_grad(loss),
    loss          = loss,
    optimizer     = make_optimizer(0.8),
    num_epochs    = num_epochs
  )

  accuracy = test_model(test_d, model).result()

  print("Test set accuracy: {:.3%}".format(accuracy))
