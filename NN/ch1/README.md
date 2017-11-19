# Handwriting recognition

I am playing here with the data and example provided by in Michael Nielson's
book. The here is all derived from his code. I won't load the dataset here, for
that see [Nielson's repo](https://github.com/mnielsen/neural-networks-and-deep-learning).

The code here works (good job Nielson!).

Here are the commands I run in a Python2 shell:

``` py
# load the data
training_data, validation_data, test_data = nist_loader.load_data_wrapper()

# initialize the random network
net = handwriting.Network([784, 30, 10])

# train the model
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```

My first run resulted in:

```
Epoch 0: 9089 / 10000
Epoch 1: 9270 / 10000
Epoch 2: 9327 / 10000
Epoch 3: 9331 / 10000
Epoch 4: 9376 / 10000
Epoch 5: 9390 / 10000
Epoch 6: 9419 / 10000
Epoch 7: 9432 / 10000
Epoch 8: 9451 / 10000
Epoch 9: 9458 / 10000
Epoch 10: 9451 / 10000
Epoch 11: 9458 / 10000
Epoch 12: 9486 / 10000
Epoch 13: 9474 / 10000
Epoch 14: 9479 / 10000
Epoch 15: 9487 / 10000
Epoch 16: 9490 / 10000
Epoch 17: 9500 / 10000
Epoch 18: 9484 / 10000
Epoch 19: 9517 / 10000
Epoch 20: 9502 / 10000
Epoch 21: 9511 / 10000
Epoch 22: 9491 / 10000
Epoch 23: 9516 / 10000
Epoch 24: 9497 / 10000
Epoch 25: 9511 / 10000
Epoch 26: 9506 / 10000
Epoch 27: 9502 / 10000
Epoch 28: 9503 / 10000
Epoch 29: 9508 / 10000
```

Of course, all I've managed to do is run his code. A few additional things
I would like to do:

 * visualize weights and biases as the training progresses
 
 * reimplement the code in Haskell

 * think through the math

 * try to apply my code to a new problem.

 * experiment with
 
   - an additional layer

   - an online method (batch size of 1)

   - a different cost function (e.g. absolute rather than square)

   - an alternative to the sigmoid function (maybe something triangular)

   - dynamic adjustment of the learning rate (e.g. if a step increases C,
     decrease the learning rate, and vice versa). Probably Nielson with discuss
     something like this in future chapters, but I would prefer to take a crack
     at the problem on my own first.
